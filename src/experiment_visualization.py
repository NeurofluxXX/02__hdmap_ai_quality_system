import pandas as pd
import matplotlib.pyplot as plt

comparison_df = pd.read_csv("../outputs/model_comparison.csv")
mlp_df = pd.read_csv("../outputs/mlp_result.csv")

all_results = pd.concat([comparison_df,mlp_df],ignore_index=True)

print("\n========== All Model Results ==========")
print(all_results)

plt.figure(figsize=(8,5))
plt.bar(all_results["model"],all_results["f1"])
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("HD Map Anomaly Detection - Model Comparison")
plt.ylim([0,1.05])
plt.tight_layout()
plt.savefig("../outputs/model_f1_comparison.png",dpi=300)
plt.close()

loss_df = pd.read_csv("../outputs/mlp_loss_history.csv")
plt.figure(figsize=(8,5))
plt.plot(loss_df["epoch"],loss_df["loss"])
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("MLP Training Loss")
plt.tight_layout()
plt.savefig("../outputs/mlp_loss_curve.png",dpi=300)
plt.close()

importance_df = pd.read_csv("../outputs/rf_feature_importance.csv")
plt.figure(figsize=(9,5))
plt.bar(importance_df["feature"],importance_df["importance"])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.savefig("../outputs/rf_feature_importance.png",dpi=300)
plt.close()