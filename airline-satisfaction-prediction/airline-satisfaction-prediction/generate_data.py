"""
generate_data.py

Generates a SYNTHETIC airline passenger satisfaction dataset that mirrors
the schema of the well-known Kaggle "Airline Passenger Satisfaction" dataset
(https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).

This project was originally built against that real dataset. Since the raw
CSV isn't bundled in this repo (and this environment has no internet access
to re-download it), this script fabricates a dataset with the SAME COLUMN
NAMES and a plausible, hand-designed relationship between features and the
target so that the rest of the pipeline (EDA, preprocessing, modeling) can
be demonstrated end-to-end.

To reproduce results on the REAL data: download the Kaggle CSV, rename it to
`data/airline_passenger_satisfaction.csv` (or point src/data_preprocessing.py
at it), and simply skip this script.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 3500

OUT_PATH = "data/airline_passenger_satisfaction.csv"


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Demographics -----------------------------------------------------
    gender = rng.choice(["Male", "Female"], size=n_rows)
    customer_type = rng.choice(
        ["Loyal Customer", "disloyal Customer"], size=n_rows, p=[0.82, 0.18]
    )
    age = rng.normal(loc=40, scale=15, size=n_rows).clip(7, 85).round().astype(int)

    type_of_travel = rng.choice(
        ["Business travel", "Personal Travel"], size=n_rows, p=[0.69, 0.31]
    )

    # Class is correlated with travel type (business travel skews Business/Eco Plus)
    travel_class = np.empty(n_rows, dtype=object)
    for i in range(n_rows):
        if type_of_travel[i] == "Business travel":
            travel_class[i] = rng.choice(
                ["Business", "Eco Plus", "Eco"], p=[0.55, 0.15, 0.30]
            )
        else:
            travel_class[i] = rng.choice(
                ["Business", "Eco Plus", "Eco"], p=[0.10, 0.15, 0.75]
            )

    flight_distance = rng.gamma(shape=2.2, scale=650, size=n_rows).clip(50, 5000).round().astype(int)

    # ---- Service rating features (0-5 Kaggle-style Likert scale) ----------
    # Draw a latent "airline quality" factor per passenger so ratings correlate
    # with each other (not fully independent), which is realistic.
    quality_latent = rng.normal(0, 1, size=n_rows)

    def rating(base_mean, noise=0.8):
        raw = base_mean + 0.9 * quality_latent + rng.normal(0, noise, size=n_rows)
        return np.clip(np.round(raw), 0, 5).astype(int)

    inflight_wifi_service = rating(2.6)
    departure_arrival_time_convenient = rating(2.9)
    ease_of_online_booking = rating(2.7)
    gate_location = rating(2.9, noise=1.0)
    food_and_drink = rating(2.9)
    online_boarding = rating(2.9)
    seat_comfort = rating(3.0)
    inflight_entertainment = rating(2.9)
    onboard_service = rating(3.1)
    leg_room_service = rating(2.9)
    baggage_handling = rating(3.2, noise=0.8)
    checkin_service = rating(3.0)
    inflight_service = rating(3.2, noise=0.8)
    cleanliness = rating(3.1)

    # Business class nudges comfort/entertainment/service ratings upward
    business_mask = travel_class == "Business"
    for arr in [seat_comfort, inflight_entertainment, onboard_service, leg_room_service, cleanliness, food_and_drink]:
        arr[business_mask] = np.clip(arr[business_mask] + 1, 0, 5)

    # ---- Delays -------------------------------------------------------------
    # Most flights have zero/near-zero delay; a right-skewed tail for the rest.
    departure_delay = rng.exponential(scale=12, size=n_rows)
    departure_delay = np.where(rng.random(n_rows) < 0.55, 0, departure_delay)
    departure_delay = departure_delay.round().astype(int).clip(0, 500)

    # Arrival delay correlated with departure delay plus its own noise
    arrival_delay = (departure_delay * rng.uniform(0.7, 1.15, size=n_rows)
                      + rng.exponential(scale=4, size=n_rows) * (rng.random(n_rows) < 0.3))
    arrival_delay = arrival_delay.round().astype(int).clip(0, 500)

    # ---- Build satisfaction target via a latent utility score --------------
    service_cols = {
        "Inflight wifi service": inflight_wifi_service,
        "Departure/Arrival time convenient": departure_arrival_time_convenient,
        "Ease of Online booking": ease_of_online_booking,
        "Gate location": gate_location,
        "Food and drink": food_and_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
    }
    avg_service_rating = np.mean(list(service_cols.values()), axis=0)

    utility = (
        1.35 * (avg_service_rating - 2.5)           # overall service quality, centered
        + 0.55 * (online_boarding - 2.5)             # online boarding matters a lot in real dataset
        + 0.45 * (inflight_wifi_service - 2.5)
        + 0.45 * (business_mask.astype(float))       # business class -> more satisfaction
        + 0.30 * (type_of_travel == "Business travel").astype(float)
        + 0.20 * (customer_type == "Loyal Customer").astype(float)
        - 0.05 * departure_delay
        - 0.04 * arrival_delay
        - 0.15 * (flight_distance < 300).astype(float)  # short annoying hops
        + rng.normal(0, 1.0, size=n_rows)             # irreducible noise
    )

    # Convert utility -> probability via logistic function, then sample label
    prob_satisfied = 1 / (1 + np.exp(-(utility - np.median(utility)) / 0.9))
    satisfied = rng.random(n_rows) < prob_satisfied
    satisfaction = np.where(satisfied, "satisfied", "neutral or dissatisfied")

    df = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "Gender": gender,
            "Customer Type": customer_type,
            "Age": age,
            "Type of Travel": type_of_travel,
            "Class": travel_class,
            "Flight Distance": flight_distance,
            **service_cols,
            "Departure Delay in Minutes": departure_delay,
            "Arrival Delay in Minutes": arrival_delay.astype(float),
            "satisfaction": satisfaction,
        }
    )

    # ---- Inject a small amount of realistic missingness ---------------------
    # Real Kaggle dataset has missing values in "Arrival Delay in Minutes".
    missing_idx = rng.choice(n_rows, size=int(0.015 * n_rows), replace=False)
    df.loc[missing_idx, "Arrival Delay in Minutes"] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated synthetic dataset with {len(df)} rows -> {OUT_PATH}")
    print(df["satisfaction"].value_counts(normalize=True).rename("proportion"))
