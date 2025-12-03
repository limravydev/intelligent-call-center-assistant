import pandas as pd
import matplotlib.pyplot as plt

# 1. Load evaluation results
df = pd.read_csv("eval_results.csv")

# 2. Per-category accuracy
cat_acc = df.groupby("category")["passed"].mean() * 100

print(cat_acc)

# 3. Bar chart
plt.figure()
cat_acc.plot(kind="bar")
plt.ylabel("Accuracy (%)")
plt.title("RAG Evaluation Accuracy by Category")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()