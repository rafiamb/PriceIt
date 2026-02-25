# PriceIt 
### Airbnb Pricing Intelligence for Chicago Hosts

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189ABD?logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=black)

Airbnb hosts often struggle to determine the optimal nightly price for their listing. Price too high and you lose bookings. Price too low and you leave money on the table.

PriceIt predicts a competitive nightly rate for Airbnb properties based on location, amenities, and property features, helping hosts maximize revenue and stay competitive.

---

## About

Built as a portfolio project to demonstrate end-to-end data science skills, from web scraping and feature engineering to model training and deployment.

---

## Demo

🔗 **Live Demo: [https://priceit.onrender.com](https://priceit.onrender.com)**

> Note: Hosted on Render free tier — first load may take 30-60 seconds to wake up.

---

## Model Performance

Trained on **7,740 Chicago Airbnb listings** scraped from Inside Airbnb. Listings above $528/night (top 5%) were excluded as they represent luxury properties outside the scope of typical hosts.

| Metric | Value |
|--------|-------|
| Test R² | 0.739 |
| MAPE | 25.3% |
| Median Absolute Error | $24.18 |
| Mean Absolute Error | $39.92 |

> The model predicts nightly rates within **$24 of market price** for a typical Chicago listing.

### Why Median Absolute Error?

Mean Absolute Error ($39.92) and RMSE ($62.27) are both inflated by luxury outliers in the dataset, specifically listings priced at $500–$528/night where even a good prediction can be off by a large dollar amount. Median Absolute Error ($24.18) is the most honest metric for this use case because it reflects what a typical host at the median price of $152/night would actually experience. MAPE (25.3%) provides additional context as a percentage-based metric that is scale-independent.

---

## Features

- Predicts nightly Airbnb price based on property details and amenities
- Displays a competitive price range based on model error
- Covers 75 Chicago neighborhoods
- Supports 10 property types
- 11 amenity flags including pool, hot tub, WiFi, gym, and more
- Clean, responsive UI built for non-technical hosts

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | XGBoost |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Data | Inside Airbnb (web scraped) |
| Deployment | Render |

---

## Project Structure

```
PriceIt/
├── main.py                  # FastAPI backend and API endpoints
├── requirements.txt         # Python dependencies
├── notebooks/
│   └── priceit.ipynb        # Data cleaning, EDA, feature engineering, model training
├── static/
│   └── index.html           # Frontend UI
└── model/
    ├── priceit_model.pkl    # Trained XGBoost model
    ├── feature_columns.pkl  # Feature column order
    ├── neighborhoods.pkl    # Chicago neighborhood list
    └── property_types.pkl   # Property type list
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/PriceIt.git
cd PriceIt
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Mac users — install OpenMP (required for XGBoost)**
```bash
brew install libomp
```

**4. Start the server**
```bash
uvicorn main:app --reload
```

**5. Open your browser at** `http://localhost:8000`

---

## Future Improvements

- Scrape and add data for additional cities beyond Chicago
- Add seasonality features to account for peak travel periods
- Include review score features for established hosts
- Add a map visualization showing comparable listings in the area
- Allow hosts to compare their current price against the model's recommendation

---

## Data

### Source
Data was sourced from [Inside Airbnb](https://insideairbnb.com), a publicly available dataset of Airbnb listings scraped from the platform. The Chicago dataset was scraped on **September 22, 2025** and originally contained **8,660 listings**.

### Cleaning
The raw dataset had significant data quality issues — not because values were missing on Airbnb itself, but due to scraping errors that produced faulty or missing values.

**Handling missing prices:**
- 835 listings had missing price values and were dropped, leaving **7,825 listings**

**Handling missing bathrooms:**
- 785 listings were missing numeric bathroom values
- These were recovered by parsing the `bathrooms_text` column (e.g. "1 bath", "Shared half-bath") and converting strings to numbers
- 13 listings where bathroom text was also missing were dropped

**Handling missing bedrooms:**
- 6 listings had missing bedroom values and were dropped as they were a negligible portion of the data

**Handling faulty scraping:**
- Many listings incorrectly showed 0 beds due to scraping errors — `beds` was excluded from modeling entirely (see Feature Engineering)

**Handling outliers:**
- Listings above **$528/night** (95th percentile) were excluded — luxury properties created a right skew and produced poor predictions for typical listings, which are the target users of this tool
- Final training set: **7,740 listings**

**Price transformation:**
- Price was log-transformed to produce a more normal distribution, which improves model performance on skewed targets

---

### Feature Engineering
Features were deliberately chosen based on what a **brand new host** would know before publishing their listing — review scores, number of reviews, and superhost status were excluded because a new host has none of that data yet.

**Location:**
- `neighbourhood_cleansed` — one-hot encoded across 75 Chicago neighborhoods to capture price differences between wealthier and less wealthy areas
- Latitude and longitude were tested but had low correlation with price (~0.13–0.14) so were excluded in favor of neighborhood

**Property:**
- `property_type` — one-hot encoded across top 10 types (rare types grouped into "Other"). Used instead of `room_type` because room type was too broad — property type had stronger signal including some types with a negative correlation with log price
- `instant_bookable` — binary encoded, affects booking conversion and competitiveness
- `minimum_nights` — captures listing positioning (short-stay vs long-stay)

**Size and comfort:**
- `bedrooms`, `bathrooms`, `accommodates` — core size features
- `beds` was excluded despite being available — it was highly correlated with bedrooms (>0.8) and had too many missing values due to scraping errors
- `bathrooms_per_guest` — engineered ratio to capture bathroom comfort relative to capacity
- `guests_per_bedroom` — engineered ratio to capture how cramped a listing feels
- `guests_per_night` — `accommodates / minimum_nights`, captures that some seemingly expensive listings are actually cheap per person for longer stays

**Amenities:**
- `amenities_count` — total amenity count had a 0.35 correlation with log price, used as a proxy for overall listing quality
- 11 high-impact amenities individually one-hot encoded: pool, hot tub, A/C, washer, dryer, gym, balcony, kitchen, WiFi, lake access, city view

> Excluding review-based features was a deliberate product decision — this tool is designed for hosts setting up a new listing who have no review history yet.

---

## License
This project is licensed under the MIT License.
