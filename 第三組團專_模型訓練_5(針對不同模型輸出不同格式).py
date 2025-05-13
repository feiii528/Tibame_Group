import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import pickle
import traceback

# 分割訓練集和測試集
def prepare_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# 根據指定閾值評估分類模型
def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    # 獲取預測概率（如果模型支持）
    if model_name == "神經網絡分類":
        # 將數據轉換為 PyTorch 張量
        X_test_tensor = torch.FloatTensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
        model.eval()  # 設置模型為評估模式
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).numpy().flatten()
        y_pred = (y_pred_proba >= threshold).astype(int)
    elif hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = None

    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 如果有概率輸出才計算 AUC
    auc_score = None
    if y_pred_proba is not None:
        auc_score = roc_auc_score(y_test, y_pred_proba)

    # 輸出評估結果
    print(f"\n{model_name} 評估結果:")
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分數: {f1:.4f}")
    if auc_score is not None:
        print(f"AUC: {auc_score:.4f}")

    # 返回結果字典
    result = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "model_object": model
    }
    return result

# 模型訓練功能
def train_random_forest(X_train, y_train, params=None):
    default_params = {
        'class_weight': 'balanced',
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 2,
        'random_state': 42
    }

    if params:
        default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, params=None):
    default_params = {
        'scale_pos_weight': 10,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 2,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, params=None):
    default_params = {
        'class_weight': 'balanced',
        'learning_rate': 0.05,
        'n_estimators': 200,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    if params:
        default_params.update(params)

    model = LGBMClassifier(**default_params)
    model.fit(X_train, y_train)
    return model

def train_svc(X_train, y_train, params=None):
    default_params = {
        'C': 10,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42
    }

    if params:
        default_params.update(params)

    model = SVC(**default_params)
    model.fit(X_train, y_train)
    return model

class BankruptcyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rates):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # 輸入層
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rates[0]))
        
        # 隱藏層
        for i in range(len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())
            if i < len(dropout_rates)-1:
                self.layers.append(nn.Dropout(dropout_rates[i+1]))
        
        # 輸出層
        self.layers.append(nn.Linear(hidden_layers[-1], 1))
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_neural_network(X_train, y_train, X_val=None, y_val=None, params=None):
    # 預設參數
    default_params = {
        'hidden_layers': [128, 64, 32],
        'dropout_rates': [0.3, 0.2, 0],
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32,
        'pos_weight': 10.0
    }

    if params:
        default_params.update(params)

    # 創建數據集
    train_dataset = BankruptcyDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=default_params['batch_size'], shuffle=True)
    
    if X_val is not None and y_val is not None:
        val_dataset = BankruptcyDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=default_params['batch_size'])

    # 創建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(X_train.shape[1], default_params['hidden_layers'], default_params['dropout_rates'])
    model = model.to(device)

    # 定義損失函數和優化器
    criterion = nn.BCELoss(weight=torch.tensor([default_params['pos_weight']]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=default_params['learning_rate'])

    # 訓練模型
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(default_params['epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # 驗證
        if X_val is not None and y_val is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs.squeeze(), batch_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                model.load_state_dict(best_model_state)
                break

    return model

# 測試不同閾值並找出最佳的
def find_optimal_threshold(model, X_test, y_test, model_name, start=0.1, end=0.9, step=0.1):
    thresholds = np.arange(start, end + step, step)
    results = []

    for threshold in thresholds:
        result = evaluate_model(model, X_test, y_test, model_name, threshold)
        result['threshold'] = threshold
        results.append(result)

    # 根據F1分數找出最佳閾值
    best_result = max(results, key=lambda x: x['f1'])

    print(f"\n最佳閾值（基於F1分數）: {best_result['threshold']:.2f}")
    print(f"對應的F1分數: {best_result['f1']:.4f}")

    return results, best_result

# 繪製分類模型比較圖表
def plot_classification_comparison(results):
    models = [result['model_name'] for result in results]
    accuracy_scores = [result['accuracy'] for result in results]
    f1_scores = [result['f1'] for result in results]
    auc_scores = [result['auc'] for result in results if result['auc'] is not None]
    auc_models = [result['model_name'] for result in results if result['auc'] is not None]

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # 準確率比較圖
    ax[0].bar(models, accuracy_scores, color='skyblue')
    ax[0].set_title('各模型準確率比較', fontsize=14)
    ax[0].set_xlabel('模型', fontsize=12)
    ax[0].set_ylabel('準確率', fontsize=12)
    ax[0].set_ylim([0, 1])
    ax[0].set_xticklabels(models, rotation=45, ha='right')
    ax[0].grid(True, alpha=0.3)

    # 為每個柱狀圖添加準確率值標籤
    for i, v in enumerate(accuracy_scores):
        ax[0].text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

    # F1分數比較圖
    ax[1].bar(models, f1_scores, color='salmon')
    ax[1].set_title('各模型F1分數比較', fontsize=14)
    ax[1].set_xlabel('模型', fontsize=12)
    ax[1].set_ylabel('F1分數', fontsize=12)
    ax[1].set_ylim([0, 1])
    ax[1].set_xticklabels(models, rotation=45, ha='right')
    ax[1].grid(True, alpha=0.3)

    # 為每個柱狀圖添加F1分數值標籤
    for i, v in enumerate(f1_scores):
        ax[1].text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

    # AUC比較圖 (如果有)
    if auc_scores:
        ax[2].bar(auc_models, auc_scores, color='lightgreen')
        ax[2].set_title('各模型AUC比較', fontsize=14)
        ax[2].set_xlabel('模型', fontsize=12)
        ax[2].set_ylabel('AUC', fontsize=12)
        ax[2].set_ylim([0, 1])
        ax[2].set_xticklabels(auc_models, rotation=45, ha='right')
        ax[2].grid(True, alpha=0.3)

        # 為每個柱狀圖添加AUC值標籤
        for i, v in enumerate(auc_scores):
            ax[2].text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

# 繪製ROC曲線
def plot_roc_curves(results, X_test, y_test):
    plt.figure(figsize=(10, 8))

    for result in results:
        model_name = result['model_name']
        model = result['model_object']

        if 'probabilities' in result and result['probabilities'] is not None:
            y_pred_proba = result['probabilities']
        elif model_name == "神經網絡分類":
            y_pred_proba = model.predict(X_test).flatten()
        elif hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            print(f"跳過 {model_name}，因為無法獲取預測概率")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')

    # 添加隨機猜測的參考線
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

# 分析特徵重要性
def analyze_feature_importance(models, feature_names):
    # 檢查哪些模型支持特徵重要性
    model_importances = {}

    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            model_importances[model_name] = pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)

    # 繪製特徵重要性圖表
    if model_importances:
        n_models = len(model_importances)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 5 * n_models))

        if n_models == 1:
            axes = [axes]

        for i, (model_name, importances) in enumerate(model_importances.items()):
            importances.head(10).plot.barh(ax=axes[i])
            axes[i].set_title(f'{model_name}-Top 10 Important Features')
            axes[i].set_xlabel('Importance Value')

        plt.tight_layout()
        plt.show()

    return model_importances

# 執行
def run_models(X_train, y_train, X_test, y_test, threshold=0.5):
    # 記錄結果
    results = []
    models = {}

    # 訓練和評估隨機森林分類器
    print("\n訓練隨機森林分類器...")
    start_time = time.time()
    rf_model = train_random_forest(X_train, y_train)
    rf_time = time.time() - start_time
    models['Random Forest'] = rf_model
    rf_result = evaluate_model(rf_model, X_test, y_test, "Random Forest", threshold)
    rf_result['training_time'] = rf_time
    results.append(rf_result)

    # 訓練和評估XGBoost分類器
    print("\n訓練XGBoost分類器...")
    start_time = time.time()
    xgb_model = train_xgboost(X_train, y_train)
    xgb_time = time.time() - start_time
    models['XGBoost'] = xgb_model
    xgb_result = evaluate_model(xgb_model, X_test, y_test, "XGBoost Classification", threshold)
    xgb_result['training_time'] = xgb_time
    results.append(xgb_result)

    # 訓練和評估LightGBM分類器
    print("\n訓練LightGBM分類器...")
    start_time = time.time()
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_time = time.time() - start_time
    models['LightGBM'] = lgb_model
    lgb_result = evaluate_model(lgb_model, X_test, y_test, "LightGBM Classification", threshold)
    lgb_result['training_time'] = lgb_time
    results.append(lgb_result)

    # 訓練和評估SVC分類器
    print("\n訓練SVC分類器...")
    start_time = time.time()
    svc_model = train_svc(X_train, y_train)
    svc_time = time.time() - start_time
    models['SVC'] = svc_model
    svc_result = evaluate_model(svc_model, X_test, y_test, "SVC Classification", threshold)
    svc_result['training_time'] = svc_time
    results.append(svc_result)

    # 為神經網絡準備驗證集
    X_train_nn, X_val, y_train_nn, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 訓練和評估神經網絡分類器
    print("\n訓練神經網絡分類器...")
    start_time = time.time()
    nn_model = train_neural_network(X_train_nn, y_train_nn, X_val, y_val)
    nn_time = time.time() - start_time
    models['Neural Network'] = nn_model
    nn_result = evaluate_model(nn_model, X_test, y_test, "神經網絡分類", threshold)
    nn_result['training_time'] = nn_time
    results.append(nn_result)

    # 打印訓練時間對比
    print("\n各模型訓練時間對比:")
    for result in results:
        print(f"{result['model_name']}: {result['training_time']:.2f} 秒")

    # 找出最佳模型 (基於F1分數)
    best_result = max(results, key=lambda x: x['f1'])
    print(f"\n最佳模型是: {best_result['model_name']}, F1 = {best_result['f1']:.4f}")

    return models, results

# 主程式
def main():
    try:
        # 加載數據
        train_data = pd.read_csv("C:/Users/ASUS/Desktop/online/專題/團體/Bankrupt/切割資料集/train_data_0314_rn.csv")
        test_data = pd.read_csv("C:/Users/ASUS/Desktop/online/專題/團體/Bankrupt/切割資料集/test_data_0314_rn.csv")

        # 準備數據
        X_train = train_data.drop("Bankrupt", axis=1)
        y_train = train_data["Bankrupt"]
        X_test = test_data.drop("Bankrupt", axis=1)
        y_test = test_data["Bankrupt"]

        # 運行模型
        models, results = run_models(X_train, y_train, X_test, y_test)

        # 找到最佳模型
        best_model = max(results, key=lambda x: x['f1'])
        best_model_name = best_model['model_name']
        best_model_object = best_model['model_object']
        
        # 創建輸出目錄
        output_dir = "C:/Users/ASUS/Desktop/online/專題/團體/Bankrupt"
        os.makedirs(output_dir, exist_ok=True)

        # 儲存最佳模型 (根據不同模型使用對應的儲存方式)
        model_path = os.path.join(output_dir, f"best_model_{best_model_name}")

        if isinstance(best_model_object, XGBClassifier):
            model_path += ".json"
            best_model_object.save_model(model_path)  # XGBoost
        elif isinstance(best_model_object, LGBMClassifier):
            model_path += ".txt"
            best_model_object.booster_.save_model(model_path)  # LightGBM
        elif "Neural Network" in best_model_name:  # 假設你的 NN 是 PyTorch
            model_path += ".pt"
            torch.save(best_model_object.state_dict(), model_path)  # PyTorch
        elif isinstance(best_model_object, (SVC, RandomForestClassifier)):
            model_path += ".pkl"
            with open(model_path, 'wb') as file:
                pickle.dump(best_model_object, file)  # SVC & RandomForest 用 pickle
        else:
            model_path += ".pkl"
            with open(model_path, 'wb') as file:
                pickle.dump(best_model_object, file)  # 其他模型用 pickle
        
        print(f"\n最佳模型 ({best_model_name}) 已儲存至 {model_path}")

        # 儲存閾值資訊
        threshold_info = {
            'model_name': best_model_name,
            'best_threshold': best_model['threshold'],
            'accuracy': best_model['accuracy'],
            'precision': best_model['precision'],
            'recall': best_model['recall'],
            'f1': best_model['f1'],
            'auc': best_model['auc']
        }
        
        threshold_path = os.path.join(output_dir, "best_threshold_info.pkl")
        with open(threshold_path, 'wb') as file:
            pickle.dump(threshold_info, file)
            
        print(f"最佳閾值資訊已儲存至 {threshold_path}")

    except Exception as e:
        print("執行時發生錯誤:", e)
        print(traceback.format_exc())

if __name__ == "__main__":
    main()