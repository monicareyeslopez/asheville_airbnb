# =============================================================================
# CUSTOMER COMMENTS: VADER Sentiment + LDA Topic Modeling
# Paste this section after your "Customer Comments" markdown cell
# Assumes: reviews, listings DataFrames already loaded; zip_to_name dict defined
# =============================================================================

# ── 0. Install / import ──────────────────────────────────────────────────────
# Run this once if needed:
# !pip install vaderSentiment gensim pyLDAvis

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
from gensim.parsing.preprocessing import (
    STOPWORDS, preprocess_string,
    strip_tags, strip_punctuation,
    strip_numeric, strip_multiple_whitespaces,
    remove_stopwords, stem_text
)
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — VADER SENTIMENT ANALYSIS
# =============================================================================

# ── 1a. Score every review ───────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

# Work on a copy; drop rows with no text
reviews_clean = reviews.dropna(subset=["comments"]).copy()
reviews_clean["comments"] = reviews_clean["comments"].astype(str)

# Apply VADER — compound score ∈ [-1, 1]
reviews_clean["vader_compound"] = reviews_clean["comments"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

# Categorical sentiment label
def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

reviews_clean["sentiment"] = reviews_clean["vader_compound"].apply(label_sentiment)

print(f"Scored {len(reviews_clean):,} reviews")
print(reviews_clean["sentiment"].value_counts(normalize=True).map("{:.1%}".format))


# ── 1b. Merge neighborhood info ──────────────────────────────────────────────
# Map listing_id → neighbourhood_cleansed → readable name
listing_neighborhood = listings[["id", "neighbourhood_cleansed"]].copy()
listing_neighborhood["neighbourhood_cleansed"] = (
    listing_neighborhood["neighbourhood_cleansed"].astype(float).astype(int)
)
listing_neighborhood["neighborhood_name"] = (
    listing_neighborhood["neighbourhood_cleansed"].map(zip_to_name)
)

reviews_geo = reviews_clean.merge(
    listing_neighborhood,
    left_on="listing_id",
    right_on="id",
    how="left"
)


# ── 1c. Aggregate sentiment by neighborhood ──────────────────────────────────
neighborhood_sentiment = (
    reviews_geo
    .groupby("neighborhood_name")
    .agg(
        avg_sentiment=("vader_compound", "mean"),
        median_sentiment=("vader_compound", "median"),
        pct_positive=("sentiment", lambda x: (x == "Positive").mean()),
        pct_negative=("sentiment", lambda x: (x == "Negative").mean()),
        review_count=("vader_compound", "count"),
    )
    .reset_index()
    .sort_values("avg_sentiment", ascending=False)
)

print("\nSentiment by Neighborhood:")
print(neighborhood_sentiment.to_string(index=False))


# ── 1d. Visualize: sentiment by neighborhood ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: avg compound score
ax = axes[0]
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in neighborhood_sentiment["avg_sentiment"]]
bars = ax.barh(
    neighborhood_sentiment["neighborhood_name"],
    neighborhood_sentiment["avg_sentiment"],
    color=colors, edgecolor="white"
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Average VADER Sentiment Score by Neighborhood", fontsize=13, fontweight="bold")
ax.set_xlabel("Avg Compound Score  (−1 = Very Negative, +1 = Very Positive)")
ax.set_ylabel("")
for bar, val in zip(bars, neighborhood_sentiment["avg_sentiment"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)

# Right: stacked % positive / neutral / negative
ax = axes[1]
neighborhood_sentiment["pct_neutral"] = (
    1 - neighborhood_sentiment["pct_positive"] - neighborhood_sentiment["pct_negative"]
)
neigh_sorted = neighborhood_sentiment.sort_values("pct_positive", ascending=True)

ax.barh(neigh_sorted["neighborhood_name"], neigh_sorted["pct_positive"],
        color="#2ecc71", label="Positive")
ax.barh(neigh_sorted["neighborhood_name"], neigh_sorted["pct_neutral"],
        left=neigh_sorted["pct_positive"], color="#bdc3c7", label="Neutral")
ax.barh(neigh_sorted["neighborhood_name"],
        neigh_sorted["pct_negative"],
        left=neigh_sorted["pct_positive"] + neigh_sorted["pct_neutral"],
        color="#e74c3c", label="Negative")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("Review Sentiment Breakdown by Neighborhood", fontsize=13, fontweight="bold")
ax.set_xlabel("Share of Reviews")
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("sentiment_by_neighborhood.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 1e. Sentiment trend over time (market-level) ─────────────────────────────
reviews_geo["date"] = pd.to_datetime(reviews_geo["date"])
reviews_geo["year_month"] = reviews_geo["date"].dt.to_period("M")

monthly_sentiment = (
    reviews_geo
    .groupby("year_month")["vader_compound"]
    .mean()
    .reset_index()
)
monthly_sentiment["year_month"] = monthly_sentiment["year_month"].dt.to_timestamp()
monthly_sentiment = monthly_sentiment[monthly_sentiment["year_month"] >= "2019-01-01"]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(monthly_sentiment["year_month"], monthly_sentiment["vader_compound"],
        color="steelblue", linewidth=2)
ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.2, label="Positive threshold")
ax.axvline(pd.Timestamp("2020-03-01"), color="tomato", linestyle="--", linewidth=1.5)
ax.text(pd.Timestamp("2020-03-01"), monthly_sentiment["vader_compound"].max() * 0.97,
        "  COVID-19", color="tomato", fontsize=8.5, va="top")
ax.axvline(pd.Timestamp("2024-09-27"), color="darkorange", linestyle="--", linewidth=1.5)
ax.text(pd.Timestamp("2024-09-27"), monthly_sentiment["vader_compound"].max() * 0.97,
        "  Hurricane\n  Helene", color="darkorange", fontsize=8.5, va="top")
ax.set_title("Average Review Sentiment Over Time (Market Level)", fontsize=13, fontweight="bold")
ax.set_ylabel("Avg VADER Compound Score")
ax.set_xlabel("")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=30)
ax.legend()
plt.tight_layout()
plt.savefig("sentiment_trend.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# SECTION 2 — LDA TOPIC MODELING
# =============================================================================

# ── 2a. Preprocessing ────────────────────────────────────────────────────────
# Custom stopwords: Airbnb-specific filler words that inflate every topic
AIRBNB_STOPWORDS = {
    "stay", "place", "host", "airbnb", "apartment", "room", "house",
    "great", "nice", "good", "perfect", "wonderful", "amazing", "love",
    "recommend", "back", "highly", "would", "could", "well", "also",
    "clean", "comfortable", "location", "everything", "really", "just",
    "trip", "time", "night", "definitely", "us", "we", "our", "like"
}
ALL_STOPWORDS = STOPWORDS.union(AIRBNB_STOPWORDS)

CUSTOM_FILTERS = [
    strip_tags,
    strip_punctuation,
    strip_numeric,
    strip_multiple_whitespaces,
    remove_stopwords,
]

def preprocess_review(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    tokens = preprocess_string(text, CUSTOM_FILTERS)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 2]
    return tokens

print("Preprocessing reviews for LDA…")
# Sample up to 50k reviews for speed; use all if your machine can handle it
sample = reviews_clean.sample(min(50_000, len(reviews_clean)), random_state=42)
processed_docs = sample["comments"].apply(preprocess_review).tolist()
processed_docs = [doc for doc in processed_docs if len(doc) >= 3]

print(f"  {len(processed_docs):,} documents after preprocessing")


# ── 2b. Build dictionary + corpus ────────────────────────────────────────────
dictionary = corpora.Dictionary(processed_docs)
# Filter extremes: ignore tokens in <5 docs or >50% of docs
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

print(f"  Vocabulary size: {len(dictionary):,} tokens")


# ── 2c. Train LDA ────────────────────────────────────────────────────────────
N_TOPICS = 6      # 6 topics works well for Airbnb reviews; tune if needed
PASSES   = 10

print(f"Training LDA with {N_TOPICS} topics…")
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=N_TOPICS,
    passes=PASSES,
    random_state=42,
    alpha="auto",
    eta="auto"
)
print("  Done.")


# ── 2d. Print top words per topic ────────────────────────────────────────────
print("\n=== Top words per topic ===")
topic_labels = {}   # fill in after reviewing — see step 2e

for topic_id in range(N_TOPICS):
    words = lda_model.show_topic(topic_id, topn=10)
    word_str = ", ".join([w for w, _ in words])
    print(f"  Topic {topic_id}: {word_str}")


# ── 2e. LABEL YOUR TOPICS ────────────────────────────────────────────────────
# After reviewing the output above, assign a human-readable label to each topic.
# These are EXAMPLES — update based on what your model actually produces:
topic_labels = {
    0: "Location & Walkability",
    1: "Bedroom & Comfort",
    2: "Communication & Check-in",
    3: "Value for Money",
    4: "Mountain Views & Nature",
    5: "Kitchen & Amenities",
}


# ── 2f. Visualize: top-words bar chart per topic ─────────────────────────────
n_cols = 3
n_rows = int(np.ceil(N_TOPICS / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()

palette = sns.color_palette("tab10", N_TOPICS)

for topic_id in range(N_TOPICS):
    words, weights = zip(*lda_model.show_topic(topic_id, topn=8))
    ax = axes[topic_id]
    ax.barh(list(words)[::-1], list(weights)[::-1],
            color=palette[topic_id], edgecolor="white")
    label = topic_labels.get(topic_id, f"Topic {topic_id}")
    ax.set_title(f"Topic {topic_id}: {label}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Word Probability")
    ax.tick_params(axis="y", labelsize=9)

for ax in axes[N_TOPICS:]:
    ax.set_visible(False)

plt.suptitle("LDA Topic Modeling: What Guests Talk About", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("lda_topics.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 2g. Assign dominant topic to each review ─────────────────────────────────
def get_dominant_topic(bow):
    topics = lda_model.get_document_topics(bow, minimum_probability=0)
    return max(topics, key=lambda x: x[1])[0]

sample["bow"] = [dictionary.doc2bow(doc) for doc in processed_docs]
sample["dominant_topic"] = sample["bow"].apply(get_dominant_topic)
sample["topic_label"] = sample["dominant_topic"].map(topic_labels)


# ── 2h. Topic distribution overall ───────────────────────────────────────────
topic_dist = (
    sample["topic_label"]
    .value_counts(normalize=True)
    .reset_index()
    .rename(columns={"index": "topic", "topic_label": "share"})
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(topic_dist["topic_label"], topic_dist["share"],
        color=palette[:len(topic_dist)], edgecolor="white")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("What Do Guests Write About? (LDA Topic Distribution)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Share of Reviews")
plt.tight_layout()
plt.savefig("lda_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 2i. Topic mix by neighborhood ────────────────────────────────────────────
# Merge neighborhood info back in
sample_geo = sample.merge(
    listing_neighborhood[["id", "neighborhood_name"]],
    left_on="listing_id", right_on="id", how="left"
)

topic_by_neighborhood = (
    sample_geo
    .groupby(["neighborhood_name", "topic_label"])
    .size()
    .reset_index(name="count")
)
topic_pivot = (
    topic_by_neighborhood
    .pivot(index="neighborhood_name", columns="topic_label", values="count")
    .fillna(0)
)
# Normalize to row percentages
topic_pivot_pct = topic_pivot.div(topic_pivot.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(13, 6))
topic_pivot_pct.plot(
    kind="bar", stacked=True, ax=ax,
    colormap="tab10", edgecolor="white", linewidth=0.5
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("What Guests Talk About — By Neighborhood (LDA Topics)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Share of Reviews")
ax.legend(title="Topic", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("lda_by_neighborhood.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 2j. Sentiment × Topic heatmap ────────────────────────────────────────────
# Merge sentiment into the sample
sample_with_sentiment = sample.merge(
    reviews_clean[["listing_id", "comments", "vader_compound"]],
    on=["listing_id", "comments"], how="left"
)

sentiment_by_topic = (
    sample_with_sentiment
    .groupby("topic_label")["vader_compound"]
    .mean()
    .reset_index()
    .sort_values("vader_compound", ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 5))
colors_st = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sentiment_by_topic["vader_compound"]]
ax.barh(sentiment_by_topic["topic_label"], sentiment_by_topic["vader_compound"],
        color=colors_st, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Average Sentiment Score by Review Topic",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Avg VADER Compound Score")
for i, (_, row) in enumerate(sentiment_by_topic.iterrows()):
    ax.text(row["vader_compound"] + 0.002, i,
            f"{row['vader_compound']:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("sentiment_by_topic.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# SECTION 3 — SUMMARY TABLE (for slides / Word doc)
# =============================================================================

summary_table = neighborhood_sentiment[[
    "neighborhood_name", "avg_sentiment", "pct_positive", "pct_negative", "review_count"
]].copy()
summary_table.columns = ["Neighborhood", "Avg Sentiment", "% Positive", "% Negative", "# Reviews"]
summary_table["% Positive"] = summary_table["% Positive"].map("{:.1%}".format)
summary_table["% Negative"] = summary_table["% Negative"].map("{:.1%}".format)
summary_table["Avg Sentiment"] = summary_table["Avg Sentiment"].map("{:.3f}".format)
summary_table["# Reviews"]    = summary_table["# Reviews"].map("{:,}".format)

print("\n=== Neighborhood Sentiment Summary ===")
print(summary_table.to_string(index=False))
