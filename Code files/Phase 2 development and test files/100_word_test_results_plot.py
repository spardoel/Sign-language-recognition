import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("100_word_test_results_OOP2.csv")

# plt.bar(x=results["word"], height=results["attempts"])
# plt.xticks(rotation=90, ha="left")
# plt.title("Number of attempts needed to identify each word")
# plt.ylabel("Number of attempts", fontweight="bold")
# plt.xlabel("Word being signed", fontweight="bold")

# plt.show()

print(results)

print(sum(results["attempts"] == 1))

print(sum(results["attempts"] < 4))


plt.bar(
    x=results["word"][results["attempts"] > 3],
    height=results["attempts"][results["attempts"] > 3],
)
plt.xticks(rotation=90, ha="left")
plt.title("Number of attempts needed to identify each word")
plt.ylabel("Number of attempts", fontweight="bold")
plt.xlabel("Word being signed", fontweight="bold")

plt.show()

g = 1
