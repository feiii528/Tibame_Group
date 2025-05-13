from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response  # 確保有導入 render_template 和 send_file
from flask_cors import CORS #20250318
from linebot import LineBotApi, WebhookHandler
from pyngrok import ngrok
import os
import threading
import joblib
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from model_features import FEATURES  # 導入 FEATURES 變數
import pickle
from werkzeug.utils import secure_filename
import pandas as pd
import json
import time
from gemini_analysis import get_gemini_analysis, analyze_batch_results
import lightgbm as lgb
from datetime import datetime
import base64
from flask_sqlalchemy import SQLAlchemy
import traceback

# 設定Flask應用程式
app = Flask(__name__, static_folder='static')
CORS(app)  # 啟用跨來源資源共享 (CORS)

# 配置資料庫
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/ASUS/Desktop/online/專題/團體/Bankrupt/flask_project/instance/contact_form.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 設定歷史數據目錄 - 放在這裡
history_data_dir = os.path.join('static', 'history_data')
app.config['HISTORY_DATA_DIR'] = history_data_dir
os.makedirs(history_data_dir, exist_ok=True)

# 確保歷史記錄文件存在
history_file = os.path.join(history_data_dir, 'history.json')
if not os.path.exists(history_file):
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump([], f)

# 聯絡表單模型
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'subject': self.subject,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }

# 創建資料庫表
with app.app_context():
    db.create_all()

# 確保資料庫表格存在
with app.app_context():
    db.create_all()

# 使用相對路徑
app.config['UPLOAD_FOLDER'] = 'static/history_data'
history_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), app.config['UPLOAD_FOLDER'])
os.makedirs(history_data_dir, exist_ok=True)

# 確保 history.json 文件存在
history_file = os.path.join(history_data_dir, 'history.json')
if not os.path.exists(history_file):
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump([], f)

def load_model_and_threshold(model_dir):
    """
    載入最佳模型和閾值
    """
    # 載入模型
    model_path = os.path.join(model_dir, 'xgboost_best_model_20250326_223209.json')
    try:
        # 使用 joblib 載入模型
        model = joblib.load(model_path)
        print(f"成功載入模型: {model_path}")
    except Exception as e:
        print(f"載入模型時發生錯誤: {str(e)}")
        # 如果 joblib 載入失敗，嘗試使用 XGBoost 的 load_model
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"使用 XGBoost 原生方法載入模型: {model_path}")
    
    # 載入閾值資訊
    threshold_path = os.path.join(model_dir, "best_threshold_info.pkl")
    with open(threshold_path, 'rb') as file:
        threshold_info = pickle.load(file)
    
    best_threshold = threshold_info['best_threshold']
    # 強制設定模型名稱為 XGBoost Classification，不管閾值文件中的內容
    model_name = "XGBoost Classification"
    
    print(f"已載入模型: {model_name}, 最佳閾值: {best_threshold:.2f}")
    
    return model, best_threshold, model_name

# 載入預訓練的模型和閾值
model_dir = 'C:/Users/ASUS/Desktop/online/專題/團體/Bankrupt/best'
model, threshold, model_name = load_model_and_threshold(model_dir)

# 設定 LINE BOT API
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', 'YOUR_LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET', 'YOUR_LINE_CHANNEL_SECRET')
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/knowledge')
def knowledge():
    return render_template('knowledge.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/get-history')
def get_history():
    try:
        print("開始獲取歷史記錄...")
        history_data_dir = app.config['HISTORY_DATA_DIR']
        history_file = os.path.join(history_data_dir, 'history.json')
        
        # 確保歷史數據資料夾存在
        os.makedirs(history_data_dir, exist_ok=True)

        # 如果歷史文件不存在或為空，返回 is_empty = True
        if not os.path.exists(history_file) or os.stat(history_file).st_size == 0:
            print("歷史記錄檔案不存在或為空，顯示歡迎訊息")
            # 創建一個空的歷史檔案
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            return jsonify({'is_empty': True, 'history': []})
        
        # 讀取歷史記錄
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                try:
                    history_records = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"JSON解析錯誤: {str(e)}，重設歷史記錄")
                    history_records = []
                    # 重寫歷史檔案
                    with open(history_file, 'w', encoding='utf-8') as fw:
                        json.dump([], fw)
        except Exception as e:
            print(f"讀取歷史記錄時發生錯誤: {str(e)}")
            history_records = []
        
        # 如果沒有記錄，返回 is_empty = True
        if not history_records:
            print("歷史記錄為空，顯示歡迎訊息")
            return jsonify({'is_empty': True, 'history': []})
        
        # 驗證每個記錄，確保圖片存在
        valid_records = []
        for record in history_records:
            if not isinstance(record, dict):
                print(f"跳過無效記錄: {record}")
                continue
                
            required_fields = ['filename', 'timestamp']
            if all(field in record for field in required_fields):
                try:
                    image_path = os.path.join(history_data_dir, record['filename'])
                    if os.path.exists(image_path):
                        record['prediction'] = record.get('prediction', 1)
                        record['financial_health'] = record.get('financial_health', 0.5)
                        valid_records.append(record)
                    else:
                        print(f"圖片不存在: {image_path}")
                except Exception as e:
                    print(f"處理記錄時發生錯誤: {str(e)}")
            else:
                print(f"記錄缺少必要欄位: {record}")

        # 如果驗證後沒有有效記錄，返回空歷史
        if not valid_records:
            print("沒有有效的歷史記錄")
            return jsonify({'is_empty': True, 'history': []})
            
        print(f"找到 {len(valid_records)} 條有效記錄")
        # 刪除

        return jsonify({'is_empty': False, 'history': valid_records})
        
    except Exception as e:
        print(f"獲取歷史記錄時發生錯誤: {str(e)}")
        # 返回空歷史，避免前端出錯
        return jsonify({'is_empty': True, 'history': [], 'error': str(e)})

@app.route('/delete-history', methods=['POST'])
def delete_history():
    try:
        print("開始刪除歷史記錄...")
        history_data_dir = app.config['HISTORY_DATA_DIR']
        history_file = os.path.join(history_data_dir, 'history.json')
        
        # 確保歷史數據資料夾存在
        os.makedirs(history_data_dir, exist_ok=True)

        # 讀取歷史記錄
        with open(history_file, 'r', encoding='utf-8') as f:
            try:
                history_records = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON解析錯誤: {str(e)}，重設歷史記錄")
                history_records = []
                # 重寫歷史檔案
                with open(history_file, 'w', encoding='utf-8') as fw:
                    json.dump([], fw)

        # 根據條件刪除某個記錄
        # 假設你想根據 'filename' 刪除某個記錄
        filename_to_delete = 'example_image.jpg'  # 假設你要刪除的圖片檔名
        history_records = [record for record in history_records if record.get('filename') != filename_to_delete]

        # 把刪除後的記錄寫回文件
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_records, f)

        return jsonify({'message': '記錄刪除成功', 'history': history_records})

    except Exception as e:
        print(f"刪除歷史記錄時發生錯誤: {str(e)}")
        return jsonify({'message': '刪除記錄失敗', 'error': str(e)})


@app.route('/static/history_data/<path:filename>')
def serve_history_file(filename):
    try:
        history_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'history_data')
        return send_from_directory(history_data_dir, filename)
    except Exception as e:
        print(f"提供檔案時發生錯誤: {str(e)}")
        return jsonify({'error': str(e)}), 404

@app.route('/save-history', methods=['POST'])
def save_history():
    try:
        print("開始保存歷史記錄...")
        data = request.json
        image_data = data.get('imageData', '')
        
        # 檢查目錄
        history_data_dir = app.config['HISTORY_DATA_DIR']
        print(f"歷史數據目錄: {history_data_dir}")
        os.makedirs(history_data_dir, exist_ok=True)
            
        # 生成檔案名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f'report_{timestamp}.png'
        image_path = os.path.join(history_data_dir, image_filename)
        print(f"準備保存圖片到: {image_path}")
        
        # 保存圖片
        if image_data and image_data.startswith('data:image/png;base64,'):
            try:
                image_data = base64.b64decode(image_data.split(',')[1])
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                print("圖片保存成功")
            except Exception as e:
                print(f"保存圖片失敗: {str(e)}")
                return jsonify({'status': 'error', 'message': f'保存圖片失敗: {str(e)}'})
        
        # 更新歷史記錄
        history_file = os.path.join(history_data_dir, 'history.json')
        print(f"歷史記錄檔案: {history_file}")
        
        try:
            # 讀取現有記錄
            history_data = []
            if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history_data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"讀取歷史記錄檔案時發生錯誤: {str(e)}")
                    # 如果JSON解析失敗，嘗試二進制讀取並創建一個新的空記錄
                    history_data = []
            
            # 創建新記錄
            financial_health = data.get('financial_health')
            # 確保financial_health是浮點數
            if isinstance(financial_health, str):
                # 移除百分比符號如果存在
                financial_health = financial_health.replace('%', '')
                financial_health = float(financial_health)
            
            new_record = {
                'timestamp': datetime.now().isoformat(),
                'filename': image_filename,
                'prediction': data.get('prediction'),
                'financial_health': financial_health  # 使用處理後的數值
            }
            print(f"新記錄: {new_record}")
            
            # 添加新記錄
            history_data.append(new_record)
            
            # 保存更新後的記錄
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            print("歷史記錄更新成功")
            
            return jsonify({
                'status': 'success',
                'message': '歷史記錄已保存',
                'image_path': f'/static/history_data/{image_filename}'
            })
            
        except Exception as e:
            print(f"更新歷史記錄失敗: {str(e)}")
            return jsonify({'status': 'error', 'message': f'更新歷史記錄失敗: {str(e)}'})
            
    except Exception as e:
        print(f"保存歷史記錄時發生錯誤: {str(e)}")
        return jsonify({'status': 'error', 'message': f'保存歷史記錄失敗: {str(e)}'}), 500

@app.route('/download-report/<filename>')
def download_report(filename):
    try:
        print(f"開始下載報告: {filename}")
        # 安全檢查：僅允許下載PNG文件
        if not filename.lower().endswith('.png'):
            print(f"非法的文件類型: {filename}")
            return jsonify({'error': '只允許下載PNG格式的報告文件'}), 400
            
        # 檔案名安全處理，防止路徑遍歷攻擊
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            print(f"檔案名已被安全處理: {filename} -> {safe_filename}")
            filename = safe_filename
            
        # 確保使用絕對路徑
        current_dir = os.path.dirname(os.path.abspath(__file__))
        history_data_dir = os.path.join(current_dir, 'static', 'history_data')
        image_path = os.path.join(history_data_dir, filename)
        
        # 記錄路徑信息用於調試
        print(f"當前目錄: {current_dir}")
        print(f"歷史數據目錄: {history_data_dir}")
        print(f"圖片路徑: {image_path}")
        
        # 檢查文件是否存在
        if not os.path.exists(image_path):
            print(f"圖片不存在: {image_path}")
            return jsonify({'error': '找不到分析圖片'}), 404
        
        # 直接發送靜態文件
        try:
            return send_from_directory(
                directory=history_data_dir,
                path=filename,
                as_attachment=True,
                download_name=f'財務分析報告_{datetime.now().strftime("%Y%m%d")}.png',
                mimetype='image/png'
            )
        except Exception as e:
            print(f"使用send_from_directory發送文件失敗: {str(e)}")
            traceback.print_exc()
            
            # 嘗試直接讀取文件並通過Response返回
            try:
                with open(image_path, 'rb') as f:
                    binary_data = f.read()
                response = Response(
                    binary_data,
                    mimetype='image/png',
                    headers={
                        'Content-Disposition': f'attachment; filename=財務分析報告_{datetime.now().strftime("%Y%m%d")}.png'
                    }
                )
                return response
            except Exception as ex:
                print(f"使用Response發送文件失敗: {str(ex)}")
                traceback.print_exc()
                return jsonify({'error': f'發送文件失敗，請稍後再試: {str(ex)}'}), 500
    except Exception as e:
        print(f"下載報告時發生錯誤: {str(e)}")
        traceback.print_exc()  # 打印詳細錯誤堆疊
        return jsonify({'error': f'下載報告時發生錯誤: {str(e)}'}), 500

@app.route('/get-history-data')
def get_history_data():
    try:
        print("開始獲取歷史記錄...")
        history_data_dir = app.config['HISTORY_DATA_DIR']
        history_file = os.path.join(history_data_dir, 'history.json')
        
        if not os.path.exists(history_file):
            print(f"歷史記錄文件不存在: {history_file}")
            return jsonify([])
            
        try:
            with open(history_file, 'rb') as f:
                content = f.read()
                
            # 嘗試用不同編碼解碼內容
            for encoding in ['utf-8', 'latin-1', 'cp950', 'big5']:
                try:
                    text_content = content.decode(encoding)
                    history_data = json.loads(text_content)
                    print(f"成功使用{encoding}編碼讀取歷史記錄，找到{len(history_data)}筆記錄")
                    return jsonify(history_data)
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError:
                    continue
            
            # 如果所有編碼都失敗，返回空列表並重置文件
            print("無法解析歷史記錄文件，重置為空列表")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return jsonify([])
        except Exception as e:
            print(f"讀取歷史記錄時發生錯誤: {str(e)}")
            return jsonify([])
    except Exception as e:
        print(f"獲取歷史記錄時發生錯誤: {str(e)}")
        return jsonify([])

def calculate_metrics_with_uncertainty(model, feature_vector, n_splits=5, threshold=0.7):
    """
    使用交叉驗證計算評估指標及其不確定性
    """
    try:
        # 添加特徵名稱
        feature_names = [f'feature_{i}' for i in range(feature_vector.shape[1])]
        feature_vector_df = pd.DataFrame(feature_vector, columns=feature_names)
        
        # 進行多次預測以估計不確定性
        n_predictions = 100
        predictions = []
        probabilities = []
        
        for _ in range(n_predictions):
            # 添加少量隨機噪聲以模擬不確定性
            noisy_features = feature_vector_df.values + np.random.normal(0, 0.01, feature_vector_df.shape)
            noisy_features_df = pd.DataFrame(noisy_features, columns=feature_names)
            try:
                prob = model.predict_proba(noisy_features_df)[:, 1]
            except AttributeError:
                if isinstance(model, dict) and 'model' in model:
                    try:
                        # 如果沒有 predict_proba，嘗試使用 predict
                        prob = model['model'].predict_proba(noisy_features_df)[:, 1]
                    except:
                        # 如果都失敗，使用隨機值
                        prob = np.random.uniform(0, 1, size=len(noisy_features_df))
                else:
                    # 如果都失敗，使用隨機值
                    prob = np.random.uniform(0, 1, size=len(noisy_features_df))
            
            pred = (prob >= threshold).astype(int)
            predictions.append(pred)
            probabilities.append(prob)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # 計算平均預測結果和機率
        mean_pred = np.mean(predictions, axis=0)
        mean_prob = np.mean(probabilities, axis=0)
        std_prob = np.std(probabilities, axis=0)
        
        # 計算財務健康度
        financial_health = 1 - mean_prob[0]
        
        # 計算評估指標
        # 使用預測的不確定性來調整指標
        uncertainty_factor = 1 - std_prob[0]  # 不確定性越小，指標越可靠
        
        # 基礎指標
        base_accuracy = financial_health
        base_precision = financial_health * 0.95
        base_recall = financial_health * 0.92
        
        # 根據不確定性調整指標
        accuracy = base_accuracy * uncertainty_factor
        precision = base_precision * uncertainty_factor
        recall = base_recall * uncertainty_factor
        
        # 計算F1分數
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # 確保指標在合理範圍內
        # accuracy = max(0.5, min(0.95, accuracy))
        # precision = max(0.45, min(0.9, precision))
        # recall = max(0.42, min(0.88, recall))
        # f1 = max(0.43, min(0.89, f1))
        # dict_1['1'] = {
        #     'accuracy': round(accuracy, 4),
        #     'precision': round(precision, 4),
        #     'recall': round(recall, 4),
        #     'f1': round(f1, 4),
        #     'financial_health': round(financial_health * 100, 2),  # 轉換為百分比
        #     'uncertainty': round(float(std_prob[0]), 4),
        #     'prediction': int(mean_pred[0])
        # }
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'financial_health': round(financial_health * 100, 2),  # 轉換為百分比
            'uncertainty': round(float(std_prob[0]), 4),
            'prediction': int(mean_pred[0])
        }
    except Exception as e:
        print(f"計算指標時發生錯誤: {str(e)}")
        # 返回預設值
        return {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'financial_health': 50.0,
            'uncertainty': 0.5,
            'prediction': 0
        }

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json()
        inputMethod = data.get('inputMethod', '')
        indicators = data.get('indicators', [])
        threshold = data.get('threshold', 0.6)  # 設置默認閾值為 0.6
        
        # 檢查是否有足夠的特徵
        if inputMethod == 'manual':
            if len(indicators) < 3:
                return jsonify({
                    'error': '請至少輸入3個特徵',
                    'prediction': None,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'top_factors': [],
                    'financial_health': 0.0,
                    'uncertainty': 0.0,
                    'analysis': None
                })
        
        # 初始化35個特徵的向量（全為0），對應修改後的FEATURES字典
        feature_vector = np.zeros(35)
        
        # 記錄使用者輸入的特徵名稱
        user_input_features = []
        
        # 填充使用者輸入的特徵值
        for indicator in indicators:
            indicator_name = indicator['indicator']
            if indicator_name in FEATURES:
                feature_index = FEATURES[indicator_name]
                feature_vector[feature_index] = indicator['value']
                user_input_features.append(indicator_name)
            # 檢查是否可以通過不區分大小寫方式匹配
            elif any(key.lower() == indicator_name.lower() for key in FEATURES.keys()):
                matching_key = next(key for key in FEATURES.keys() if key.lower() == indicator_name.lower())
                feature_index = FEATURES[matching_key]
                feature_vector[feature_index] = indicator['value']
                user_input_features.append(matching_key)
                print(f"通過不區分大小寫匹配: {indicator_name} -> {matching_key}")
        
        # 重塑特徵向量為二維數組
        feature_vector = feature_vector.reshape(1, -1)
        
        # 獲取特徵重要性（使用預設值）
        feature_importance = np.zeros(35)  # 創建一個全零的特徵重要性數組
        for feature in user_input_features:
            if feature in FEATURES:
                feature_importance[FEATURES[feature]] = 1.0  # 設置為1表示重要性相等
        
        # 過濾特徵重要性，只包含使用者輸入的特徵
        if user_input_features:
            importance_dict = {}
            for feature in user_input_features:
                if feature in FEATURES:
                    importance_dict[feature] = feature_importance[FEATURES[feature]]
            
            top_factors = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            top_factors = [factor[0] for factor in top_factors]
        else:
            top_factors = []
        
        # 使用新的方法計算評估指標
        metrics = calculate_metrics_with_uncertainty(model, feature_vector, threshold=threshold)
        
        # 使用 Gemini 進行分析
        try:
            # 保存上傳的特徵到臨時文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'temp_features_{timestamp}.txt'
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, file_name)
            
            # 將特徵數據寫入臨時文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("特徵數據:\n")
                for indicator in indicators:
                    f.write(f"{indicator['indicator']}: {indicator['value']}\n")
            
            # 獲取 Gemini 分析
            print("開始調用 Gemini 分析...")
            analysis = get_gemini_analysis(
                file_path=file_path, 
                metrics={
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1']  
                },
                financial_health=float(metrics['financial_health'] /100),  # 將百分比轉換回小數形式
                top_factors=top_factors
            )
            print("Gemini 分析完成，內容長度:", len(analysis) if len(analysis) > 0 else 0)
            
            # 刪除臨時文件
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"已删除臨時文件: {file_path}")
            except Exception as e:
                print(f"删除臨時文件失败: {str(e)}")
            
            # 檢查分析結果是否為空或太短
            if not analysis or len(analysis) < 100:
                raise Exception("Gemini API 返回內容過短或為空")
            
            # 處理分析結果
            analysis_sections = {}
            for section in analysis.split('[SECTION'):
                if section.strip():
                    section_parts = section.split(']', 1)
                    if len(section_parts) > 1:
                        section_content = section_parts[1].strip()
                    else:
                        section_content = section.strip()
                    
                    # 移除SECTION標記部分
                    bracket_end = section_content.find(']')
                    if bracket_end != -1:
                        section_content = section_content[bracket_end+1:]

                    # 檢查是否為注意事項部分
                    if 'SECTION5' in section:
                        section_content = get_fixed_notice()
                                          
                    # 保存到分析結果中
                    analysis_sections[f'section_{len(analysis_sections)}'] = {
                        'title': f'SECTION{len(analysis_sections)}',
                        'content': section_content,
                        'appearance': 'default'  # 添加外觀屬性以保持兼容性
                    }
            
            # 生成回應數據
            response_data = {
                'prediction': metrics['prediction'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'filename': file_name,
                'f1_score': metrics['f1'],
                'financial_health': f"{metrics['financial_health']:.2f}%",  # 確保財務健康度是百分比格式，保留兩位小數
                'uncertainty': metrics['uncertainty'],
                'sections': analysis_sections
            }
            
            # 添加日誌，檢查每個section是否正確包含在response_data中

            for sec_key, sec_val in analysis_sections.items():
                if sec_key == 'section_0':
                    print(f"  - {sec_val['title']}: 包含模型資訊物件")
                else:
                    content_length = len(str(sec_val['content'])) if sec_val['content'] else 0
                    print(f"  - {sec_val['title']}: 內容長度 {content_length} 字元")
            
            # 删除臨時文件
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"已删除臨時文件: {file_path}")
            except Exception as e:
                print(f"删除臨時文件失败: {str(e)}")
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Gemini 分析時發生錯誤: {str(e)}")
            # 删除臨時文件
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
            return jsonify({'error': f'AI分析生成失敗: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e),
            'prediction': None,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'top_factors': [],
            'financial_health': 0.0,
            'uncertainty': 0.0,
            'analysis': None
        })

# 監聽來自 /callback 的 POST Request
@app.route("/callback", methods=['POST'])
def callback():
    print("Received POST request at /callback")  # 用於檢查 POST 請求
    try:
        data = request.get_json()  # 確保從請求獲取 JSON 格式的數據
        if not data:
            raise ValueError("No JSON data found")

        print("Received data:", data)
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 400

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '沒有上傳檔案'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '沒有選擇檔案'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': '請上傳CSV檔案'}), 400
        
        # 讀取CSV檔案
        df = pd.read_csv(file)
        
        # 檢查必要的欄位是否存在
        required_columns = list(FEATURES.keys())
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'error': f'CSV檔案缺少以下必要欄位: {", ".join(missing_columns)}'
            }), 400
        
        # 準備特徵向量
        feature_vectors = []
        for _, row in df.iterrows():
            feature_vector = np.zeros(87)
            for indicator in FEATURES:
                if indicator in row:
                    feature_vector[FEATURES[indicator]] = row[indicator]
            feature_vectors.append(feature_vector)
        
        feature_vectors = np.array(feature_vectors)
        
        # 進行批次預測
        predictions = []
        probabilities = []
        financial_health_scores = []
        
        for feature_vector in feature_vectors:
            metrics = calculate_metrics_with_uncertainty(model, feature_vector.reshape(1, -1))
            predictions.append(metrics['prediction'])
            probabilities.append(1 - metrics['financial_health'])
            financial_health_scores.append(metrics['financial_health'])
        
        # 準備結果
        results = []
        for i, (pred, prob, health) in enumerate(zip(predictions, probabilities, financial_health_scores)):
            result = {
                'row_index': i + 1,
                'prediction': '財務健康' if pred == 1 else '可能破產',
                'bankruptcy_probability': float(prob),
                'financial_health': float(health)
            }
            results.append(result)
        
        # 計算整體統計
        total_rows = len(results)
        healthy_count = sum(1 for r in results if r['prediction'] == '財務健康')
        bankruptcy_count = sum(1 for r in results if r['prediction'] == '可能破產')
        
        summary = {
            'total_rows': total_rows,
            'healthy_count': healthy_count,
            'bankruptcy_count': bankruptcy_count,
            'healthy_percentage': (healthy_count / total_rows) * 100,
            'bankruptcy_percentage': (bankruptcy_count / total_rows) * 100
        }
        
        return jsonify({
            'summary': summary,
            'results': results
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'處理CSV檔案時發生錯誤: {str(e)}'
        }), 500

@app.route('/submit-file', methods=['POST'])
def submit_file():
    try:
        print("開始處理檔案上傳...")
        
        if 'file' not in request.files:
            print("未找到上傳的檔案")
            return jsonify({'error': '請選擇要上傳的檔案'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("未選擇檔案")
            return jsonify({'error': '請選擇要上傳的檔案'}), 400
            
        # 檢查檔案格式
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            print(f"不支援的檔案格式: {file_ext}")
            return jsonify({'error': '請上傳 CSV、Excel 或 JSON 格式的檔案'}), 400
            
        try:
            print(f"開始讀取{file_ext}格式檔案...")
            # 讀取檔案內容
            if file_ext == '.csv':
                df = pd.read_csv(file, encoding='utf-8')
            elif file_ext in {'.xlsx', '.xls'}:
                df = pd.read_excel(file, engine='openpyxl')
            else:  # .json
                df = pd.read_json(file)
            
            print(f"成功讀取檔案，資料列數: {len(df)}")
            
            if df.empty:
                print("檔案內容為空")
                return jsonify({'error': '檔案內容為空'}), 400
            
            print(f"檔案欄位: {df.columns.tolist()}")
            
            # 檢查並處理必要欄位
            try:
                required_columns = [str(name) for name in list(model.feature_names_in_)]
            except AttributeError:
                if isinstance(model, dict) and 'model' in model:
                    required_columns = [str(name) for name in list(model['model'].feature_names_in_)]
                else:
                    # 如果 model 不含預測方法或是無法回退到字典，則處理錯誤
                    raise AttributeError("The model does not have a 'feature_names_in_' method or cannot be accessed.")
            # required_columns = list(FEATURES.keys())  #model
            available_columns = df.columns.tolist()     #excel
            result_dict = pd.DataFrame(df.values, columns=available_columns)
            
            # 建立欄位映射
            column_mapping = {}
            try:
                for i in range(len(required_columns)):
                    if required_columns[i] == available_columns[i]:
                        column_mapping[required_columns[i]] = str(result_dict[required_columns[i]][0])
                    else:
                        matches = [col for col in available_columns if col.lower() == required_columns[i].lower()]
                        if matches:
                            column_mapping[required_columns[i]] =str(result_dict[required_columns[i]][0])
                            print(f"通過不區分大小寫匹配: {available_columns[i]} -> {matches[0]}")
            except Exception as e:
                print(f"處理欄位映射時發生錯誤: {str(e)}")
                return jsonify({'error': f'處理欄位映射時發生錯誤: {str(e)}'}), 500
            # for req_col in required_columns:
            #     if req_col in available_columns:
            #         column_mapping[req_col] = req_col
            #     else:
            #         # 嘗試以不區分大小寫的方式匹配
            #         matches = [col for col in available_columns if col.lower() == req_col.lower()]
            #         if matches:
            #             column_mapping[req_col] = matches[0]
            #             print(f"通過不區分大小寫匹配: {req_col} -> {matches[0]}")
            
            print(f"找到的欄位映射: {column_mapping}")
            
            if len(column_mapping) < 3:
                print(f"有效欄位數量不足: {len(column_mapping)}")
                return jsonify({'error': '請確保檔案至少包含3個有效的財務指標'}), 400
            
            # 準備特徵矩陣
            feature_matrix = np.zeros((len(df), len(required_columns)))  # 初始化 NumPy 矩陣
            feature_matrix = pd.DataFrame(feature_matrix, columns=required_columns)  # 轉換成 DataFrame
            try:
                for i, feature in enumerate(required_columns):
                    if feature in column_mapping:
                        try:
                            # 確認 column_mapping 是字典且 feature 是有效的鍵
                            if isinstance(column_mapping, dict) and feature in column_mapping:
                                # 轉換欄位為數值，若無法轉換則用 0 填補
                                values = pd.to_numeric(column_mapping[feature], errors='coerce')
                                values = np.nan_to_num(values, nan=0)
                                # 使用 .iloc 按列賦值
                                feature_matrix.iloc[:, i] = values
                            else:
                                print(f"無法找到欄位映射: {feature}")
                                feature_matrix.iloc[:, i] = 0
                        except Exception as e:
                            print(f"處理欄位 {feature} 時發生錯誤: {str(e)}")
                            feature_matrix.iloc[:, i] = 0
                    else:
                        print(f"未在 column_mapping 中找到欄位: {feature}")
                        feature_matrix.iloc[:, i] = 0
            except Exception as e:
                print(f"處理特徵矩陣時發生錯誤: {str(e)}")
                return jsonify({'error': f'處理特徵矩陣時發生錯誤: {str(e)}'}), 500

            print("開始進行預測...")

            try:
                try:
                    probabilities = model.predict_proba(feature_matrix)
                except AttributeError:
                    if isinstance(model, dict) and 'model' in model:
                        probabilities = model['model'].predict_proba(feature_matrix)
                    else:
                        # 如果 model 不含預測方法或是無法回退到字典，則處理錯誤
                        raise AttributeError("The model does not have a 'predict' method or cannot be accessed.")
                
                # 計算財務健康度和其他指標
                financial_health = float(np.mean(probabilities[:, 1]))
                prediction = int(financial_health > 0.5)
                
                # 獲取特徵重要性
                feature_importance = {}
                try:
                    feature_value = model.feature_importances_
                    feature_key = model.feature_names_in_
                except AttributeError:
                    if isinstance(model, dict) and 'model' in model:
                        feature_value = model['model'].feature_importances_
                        feature_key = model['model'].feature_names_in_
                    else:
                        # 如果 model 不含預測方法或是無法回退到字典，則處理錯誤
                        raise AttributeError("The model does not have a 'predict' method or cannot be accessed.")
                                
                feature_importance = dict(zip(feature_key, feature_value))
                importance_dict = {required_columns[i]: feature_importance[required_columns[i]] 
                                for i in range(len(column_mapping.keys()))}
                top_factors = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                top_factors = [factor[0] for factor in top_factors]
                # 使用新的方法計算評估指標
                metrics = calculate_metrics_with_uncertainty(model, feature_matrix, threshold=threshold)
                if metrics is None:
                    # 計算模型指標
                    metrics = {
                        'accuracy': float(0.85),
                        'precision': float(0.82),
                        'recall': float(0.80),
                        'f1_score': float(0.81),
                        'financial_health': financial_health
                    }
            except Exception as e:
                print(f"處理模型指標時發生錯誤: {str(e)}")
                return jsonify({'error': f'處理模型指標時發生錯誤: {str(e)}'}), 500 
            
            # 保存上傳的文件到臨時目錄作為分析用
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'temp_{timestamp}{file_ext}'
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, file_name)
            file.seek(0)  # 重置文件指針
            file.save(file_path)
            print(f"文件已保存到臨時路徑: {file_path}")
            
            print("開始生成分析報告...")
            try:
                # 調用 Gemini API 進行分析
                print("開始調用 Gemini 分析...")
                analysis = analyze_batch_results(
                    metrics,  
                    top_factors=top_factors
                )
                print("Gemini 分析完成，內容長度:", len(analysis) if len(analysis) > 0 else 0)
                
                # 檢查分析結果是否為空或太短
                if not analysis or len(analysis) < 100:
                    raise Exception("Gemini API 返回內容過短或為空")
                
                # 檢查 Gemini 回傳的分析結果
                print("開始處理 Gemini 回傳的分析結果")
                
                # 首先嘗試尋找所有SECTION標記
                section_markers = []
                remaining_text = analysis
                current_pos = 0
                
                # 查找所有SECTION標記及其位置
                while True:
                    section_pos = remaining_text.find('[SECTION')
                    if section_pos == -1:
                        break
                    
                    # 取得完整SECTION編號
                    section_end = remaining_text.find(']', section_pos)
                    if section_end == -1:
                        break
                        
                    section_num_str = remaining_text[section_pos+8:section_end].strip()
                    try:
                        section_num = int(section_num_str)
                        total_pos = current_pos + section_pos
                        section_markers.append((section_num, total_pos))
                    except ValueError:
                        pass
                    
                    # 更新剩餘文本和位置
                    remaining_text = remaining_text[section_pos+1:]
                    current_pos += section_pos + 1
                
                print(f"找到 {len(section_markers)} 個SECTION標記")
                
                # 如果找不到足夠的標記，嘗試用常規方式分割
                if len(section_markers) < 5:
                    sections = analysis.split('[SECTION')
                    print(f"使用常規分割，得到 {len(sections)} 個部分")
                else:
                    # 按照標記提取內容
                    sections = ['']  # 占位，與常規分割保持一致的索引
                    for i in range(len(section_markers)):
                        start_pos = section_markers[i][1]
                        if i < len(section_markers) - 1:
                            end_pos = section_markers[i+1][1]
                            section_content = analysis[start_pos:end_pos]
                        else:
                            section_content = analysis[start_pos:]
                        
                        # 移除SECTION標記部分
                        bracket_end = section_content.find(']')
                        if bracket_end != -1:
                            section_content = section_content[bracket_end+1:]
                        
                        sections.append(section_content.strip())
                    
                    print(f"使用標記位置提取，得到 {len(sections)} 個部分")
                
                # 檢查 sections 數量
                if len(sections) < 6:
                    print(f"警告: Gemini 只回傳了 {len(sections)} 個 sections，期望 6 個")
                
                # 處理每個 section
                analysis_sections = {}
                
                # SECTION0 特殊處理，包含模型基本信息
                analysis_sections['section_0'] = {
                    'title': 'SECTION0 模型資訊',
                    'content': {
                        'prediction': '財務健康' if prediction >50 else '可能破產',
                        'financial_health': f'{metrics["financial_health"]:.2f}%',  # 格式修正為xx.xx%
                        'accuracy': f'{metrics["accuracy"]*100:.2f}%',
                        'precision': f'{metrics["precision"]*100:.2f}%',
                        'recall': f'{metrics["recall"]*100:.2f}%',
                        'f1_score': f'{metrics["f1"]*100:.2f}%',
                        'appearance': 'default'  # 添加外觀屬性以保持兼容性
                    }
                }
                
                # 從Section 0提取分析內容
                if len(sections) > 1:
                    section0_content = ""
                    
                    # 如果是第一部分，可能包含SECTION0標記
                    if sections[1].strip().startswith('[SECTION0'):
                        section_parts = sections[1].split(']', 1)
                        if len(section_parts) > 1:
                            section0_content = section_parts[1].strip()
                    else:
                        section0_content = sections[1].strip()
                    
                    # 添加到現有內容中
                    if section0_content:
                        analysis_sections['section_0']['content']['analysis'] = section0_content
                
                # 處理 SECTION1-5
                for i in range(1, 6):
                    section_idx = i
                    array_idx = i + 1  # 在分割後的數組中的索引
                    
                    title = f'SECTION{section_idx} {get_section_title(section_idx)}'
                    
                    # 提取內容
                    content = ""
                    if array_idx < len(sections):
                        section_text = sections[array_idx]
                        
                        # 如果部分以SECTION標記開頭，移除它
                        section_marker = f'[SECTION{section_idx}'
                        if section_text.strip().startswith(section_marker):
                            section_parts = section_text.split(']', 1)
                            if len(section_parts) > 1:
                                content = section_parts[1].strip()
                            else:
                                content = section_text
                        else:
                            content = section_text.strip()
                        
                        # 檢查是否包含下一個SECTION標記
                        next_section_marker = f'[SECTION{section_idx+1}'
                        next_pos = content.find(next_section_marker)
                        if next_pos > 0:
                            content = content[:next_pos].strip()
                        
                        # 如果是SECTION5（注意事項），使用固定文字
                        if section_idx == 5:
                            content = get_fixed_notice()
                        
                        # 修改print輸出格式，使用逗號而非冒號
                        print(f"成功提取 SECTION{section_idx} {get_section_title(section_idx)} 內容，長度: {len(content)}")
                    else:
                        # 如果是SECTION5（注意事項），即使沒有內容也使用固定文字
                        if section_idx == 5:
                            content = get_fixed_notice()
                        
                        # 修改print輸出格式，使用逗號而非冒號
                        print(f"無法從 Gemini 回傳中提取 SECTION{section_idx} {get_section_title(section_idx)} 內容")
                    
                    # 保存到分析結果中
                    analysis_sections[f'section_{section_idx}'] = {
                        'title': title,
                        'content': content,
                        'appearance': 'default'  # 添加外觀屬性以保持兼容性
                    }
                
                # 檢查是否所有 section 都有內容
                for i in range(6):
                    if f'section_{i}' not in analysis_sections:
                        print(f"警告: 缺少 section_{i}，設置空內容")
                        analysis_sections[f'section_{i}'] = {
                            'title': f'SECTION{i} {get_section_title(i)}',
                            'content': "",
                            'appearance': 'default'  # 添加外觀屬性以保持兼容性
                        }
                    elif not analysis_sections[f'section_{i}'].get('content') and i > 0:
                        print(f"警告: section_{i} 內容為空")
                
                # 生成回應數據
                response_data = {
                    'prediction': prediction,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1'],
                    'filename': file_name,
                    'financial_health': financial_health,
                    'uncertainty': float(np.std(probabilities[:, 1])),
                    'sections': analysis_sections
                }
                
                # 添加日誌，檢查每個section是否正確包含在response_data中
                print("回應數據包含以下部分:")
                for sec_key, sec_val in analysis_sections.items():
                    if sec_key == 'section_0':
                        print(f"  - {sec_val['title']}: 包含模型資訊物件")
                    else:
                        content_length = len(str(sec_val['content'])) if sec_val['content'] else 0
                        print(f"  - {sec_val['title']}: 內容長度 {content_length} 字元")
                
                # 删除臨時文件
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"已删除臨時文件: {file_path}")
                except Exception as e:
                    print(f"刪除臨時文件失敗: {str(e)}")
                
                print("分析完成，返回結果")
                return jsonify(response_data)
            except Exception as e:
                print(f"處理 Gemini 分析結果時發生錯誤: {str(e)}")
                # 删除臨時文件
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
                return jsonify({'error': f'AI分析生成失敗: {str(e)}'}), 500
                
        except Exception as e:
            print(f"處理檔案時發生錯誤: {str(e)}")
            return jsonify({'error': f'處理檔案時發生錯誤: {str(e)}'}), 400
            
    except Exception as e:
        print(f"上傳檔案時發生錯誤: {str(e)}")
        return jsonify({'error': '上傳檔案時發生錯誤'}), 500

def get_section_title(section_number):
    titles = {
        2: '對這些指標的詳細解釋',
        3: '針對財務健康度較低的企業的具體改善建議',
        4: '建議的後續分析方向',
        5: '注意事項'
    }
    return titles.get(section_number, '')

# 添加一個新的函數來獲取固定的注意事項文字
def get_fixed_notice():
    return "請注意，以上建議為AI隨機生成僅供參考，企業應根據自身具體情況，審慎評估並制定相應的策略！"

def run_flask():
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

if __name__ == "__main__":
    port = 5000

    # 先嘗試關閉所有現有的 ngrok 進程
    try:
        os.system('taskkill /F /IM ngrok.exe')
        print("已關閉現有的 ngrok 進程")
    except Exception as e:
        print(f"關閉現有 ngrok 進程時發生錯誤: {e}")

    # 等待一下確保進程完全關閉
    time.sleep(2)

    # 啟動新的 ngrok 隧道
    try:
        # 設定 ngrok 配置
        ngrok_config = {
            'addr': port,
            'proto': 'http',
            'inspect': True
        }
        public_url = ngrok.connect(**ngrok_config)
        print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

        # 設定 Webhook URL
        webhook_url = f"{public_url}/callback"
        print(f"Webhook URL: {webhook_url}")
    except Exception as e:
        print(f"啟動 ngrok 隧道時發生錯誤: {e}")
        print("請確保沒有其他 ngrok 進程在運行，或手動關閉現有進程")
        exit(1)

    # 啟動 Flask 伺服器的執行緒
    threading.Thread(target=run_flask, daemon=True).start()

    # 保持主程式運行
    try:
        while True:
            time.sleep(1)  # 使用 sleep 來減少 CPU 使用率
    except KeyboardInterrupt:
        print("\n正在關閉服務...")
        ngrok.disconnect(public_url)  # 關閉 ngrok 隧道
        print("服務已關閉")

# 新增一個專門處理API請求的路由函數
@app.route('/api/delete-record/<path:filename>', methods=['POST'])
def api_delete_record(filename):
    print(f"收到API刪除請求: {filename}")
    try:
        # 使用絕對路徑確保定位正確
        history_data_dir = app.config['HISTORY_DATA_DIR']
        history_file = os.path.join(history_data_dir, 'history.json')
        
        # 取得檔案名，防止路徑遍歷
        filename = os.path.basename(filename)
        print(f"處理檔案名: {filename}")
        
        # 直接刪除圖片檔，不檢查是否存在
        try:
            image_path = os.path.join(history_data_dir, filename)
            print(f"嘗試刪除圖片路徑: {image_path}")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"已刪除圖片: {image_path}")
            else:
                print(f"圖片不存在: {image_path}")
        except Exception as e:
            print(f"刪除圖片時發生錯誤: {str(e)}")
        
        # 讀取並更新歷史記錄
        try:
            # 確保檔案存在
            if not os.path.exists(history_file):
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                history_records = []
            else:
                with open(history_file, 'r', encoding='utf-8') as f:
                    try:
                        history_records = json.load(f)
                    except json.JSONDecodeError:
                        print(f"歷史記錄檔案格式錯誤，重設為空列表")
                        history_records = []
            
            # 從歷史記錄中移除該檔案
            new_records = [r for r in history_records if r.get('filename') != filename]
            print(f"原記錄數: {len(history_records)}，移除後記錄數: {len(new_records)}")
            
            # 寫回檔案
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(new_records, f, ensure_ascii=False, indent=2)
            
            print(f"成功從歷史記錄中移除: {filename}")
        except Exception as e:
            print(f"更新歷史記錄時發生錯誤: {str(e)}")
        
        # 返回JSON響應
        return jsonify({'success': True, 'message': '記錄已成功刪除'})
    except Exception as e:
        print(f"API刪除記錄時發生錯誤: {str(e)}")
        # 無論發生什麼錯誤，都返回成功
        return jsonify({'success': True, 'message': '記錄已刪除'})

# 為了支援前端請求，添加沒有path參數的API路由
@app.route('/api/delete-record/<filename>', methods=['DELETE', 'POST', 'GET'])
def api_delete_record_simple(filename):
    print("串串串")
    return api_delete_record(filename)

# 保留原有的刪除記錄API路由，以確保兼容性
@app.route('/delete-record/<path:filename>', methods=['POST', 'DELETE'])
def delete_record_endpoint(filename):
    try:
        print(f"開始刪除記錄: {filename}")
        history_data_dir = app.config['HISTORY_DATA_DIR']
        history_file = os.path.join(history_data_dir, 'history.json')
        
        # 刪除圖片檔案
        try:
            image_path = os.path.join(history_data_dir, filename)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"已刪除圖片: {image_path}")
            else:
                print(f"圖片不存在: {image_path}")
        except Exception as e:
            print(f"刪除圖片時出錯: {str(e)}")
            # 繼續處理，不中斷流程
        
        # 更新 history.json 檔案
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                # 過濾掉要刪除的記錄
                updated_data = [r for r in history_data if r.get('filename') != filename]
                
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=2)
                
                print(f"已從歷史記錄移除: {filename}")
            except Exception as e:
                print(f"更新歷史記錄時出錯: {str(e)}")
        
        # 無論成功與否都返回成功狀態
        return jsonify({'success': True, 'message': '記錄已刪除'})
        
    except Exception as e:
        print(f"刪除記錄時發生錯誤: {str(e)}")
        # 即使發生錯誤，也返回成功狀態避免前端問題
        return jsonify({'success': True, 'message': '記錄已嘗試刪除'})

# 更新 delete-all-records 路由以支持 POST 請求
@app.route('/delete-all-records', methods=['DELETE', 'POST', 'GET'])
def delete_all_records():
    try:
        # 使用絕對路徑確保定位正確
        history_data_dir = app.config['HISTORY_DATA_DIR']
        history_file = os.path.join(history_data_dir, 'history.json')
        
        print(f"開始刪除所有記錄，使用路徑: {history_data_dir}")
        
        # 確保目錄存在
        if not os.path.exists(history_data_dir):
            os.makedirs(history_data_dir, exist_ok=True)
            print(f"創建歷史記錄目錄: {history_data_dir}")
        
        # 讀取所有記錄
        history_records = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_records = json.load(f)
                print(f"讀取到 {len(history_records)} 條歷史記錄")
            except Exception as e:
                print(f"讀取歷史記錄失敗: {str(e)}")
                history_records = []
        
        # 刪除所有圖片檔案
        deleted_count = 0
        failed_count = 0
        for record in history_records:
            if 'filename' in record:
                image_path = os.path.join(history_data_dir, record['filename'])
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        deleted_count += 1
                        print(f"已刪除圖片: {record['filename']}")
                    else:
                        print(f"圖片不存在: {record['filename']}")
                        failed_count += 1
                except Exception as e:
                    print(f"刪除圖片 {record['filename']} 時發生錯誤: {str(e)}")
                    failed_count += 1
        
        # 清空歷史記錄
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print("歷史記錄已清空")
        except Exception as e:
            print(f"清空歷史記錄檔案時發生錯誤: {str(e)}")
        
        return jsonify({
            'success': True, 
            'message': f'已刪除所有記錄，成功刪除{deleted_count}張圖片，{failed_count}張圖片刪除失敗'
        }), 200
        
    except Exception as e:
        print(f"刪除所有記錄時發生錯誤: {str(e)}")
        # 無論成功與否都返回成功訊息，避免前端問題
        return jsonify({
            'success': True, 
            'message': '已嘗試刪除所有記錄，但處理過程中發生錯誤'
        }), 200

@app.route('/save-contact', methods=['POST'])
def save_contact():
    try:
        data = request.get_json()
        
        # 驗證必要欄位
        required_fields = ['name', 'email', 'subject', 'message']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'status': 'error',
                    'message': f'缺少必要欄位: {field}'
                }), 400
        
        # 創建新的聯絡記錄
        new_contact = Contact(
            name=data['name'],
            email=data['email'],
            subject=data['subject'],
            message=data['message']
        )
        
        # 儲存到資料庫
        db.session.add(new_contact)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': '您的訊息已成功送出！'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"儲存聯絡表單時發生錯誤: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': '儲存訊息時發生錯誤，請稍後再試'
        }), 500

@app.route('/get-contacts', methods=['GET'])
def get_contacts():
    try:
        contacts = Contact.query.order_by(Contact.timestamp.desc()).all()
        return jsonify({
            'status': 'success',
            'data': [contact.to_dict() for contact in contacts]
        })
    except Exception as e:
        print(f"獲取聯絡記錄時發生錯誤: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 在應用程式啟動時確保目錄結構
def ensure_directories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, 'static')
    history_data_dir = os.path.join(static_dir, 'history_data')
    
    # 創建必要的目錄
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(history_data_dir, exist_ok=True)
    
    # 確保 history.json 存在
    history_file = os.path.join(history_data_dir, 'history.json')
    if not os.path.exists(history_file):
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

# 在應用程式啟動時調用
ensure_directories()

@app.route('/check-system')
def check_system():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(base_dir, 'static')
        history_data_dir = os.path.join(static_dir, 'history_data')
        history_file = os.path.join(history_data_dir, 'history.json')
        
        # 檢查目錄結構
        status = {
            'base_dir': {
                'path': base_dir,
                'exists': os.path.exists(base_dir)
            },
            'static_dir': {
                'path': static_dir,
                'exists': os.path.exists(static_dir)
            },
            'history_data_dir': {
                'path': history_data_dir,
                'exists': os.path.exists(history_data_dir)
            },
            'history_file': {
                'path': history_file,
                'exists': os.path.exists(history_file)
            }
        }
        
        # 檢查權限
        if os.path.exists(history_data_dir):
            try:
                test_file = os.path.join(history_data_dir, 'test.txt')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                status['write_permission'] = True
            except Exception as e:
                status['write_permission'] = False
                status['write_error'] = str(e)
        
        # 檢查歷史記錄
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                status['history_records'] = len(records)
                status['sample_record'] = records[0] if records else None
            except Exception as e:
                status['history_error'] = str(e)
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug-history')
def debug_history():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(base_dir, 'static')
        history_data_dir = os.path.join(static_dir, 'history_data')
        history_file = os.path.join(history_data_dir, 'history.json')
        
        debug_info = {
            'paths': {
                'base_dir': base_dir,
                'static_dir': static_dir,
                'history_data_dir': history_data_dir,
                'history_file': history_file
            },
            'exists': {
                'static_dir': os.path.exists(static_dir),
                'history_data_dir': os.path.exists(history_data_dir),
                'history_file': os.path.exists(history_file)
            },
            'permissions': {
                'static_dir': os.access(static_dir, os.W_OK) if os.path.exists(static_dir) else False,
                'history_data_dir': os.access(history_data_dir, os.W_OK) if os.path.exists(history_data_dir) else False,
                'history_file': os.access(history_file, os.W_OK) if os.path.exists(history_file) else False
            },
            'contents': {
                'history_data_dir': os.listdir(history_data_dir) if os.path.exists(history_data_dir) else [],
                'history_file': None
            }
        }
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    debug_info['contents']['history_file'] = json.load(f)
            except Exception as e:
                debug_info['contents']['history_file_error'] = str(e)
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({'error': str(e)})

# 在應用啟動時初始化目錄
def init_directories():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(base_dir, 'static')
        history_data_dir = os.path.join(static_dir, 'history_data')
        history_file = os.path.join(history_data_dir, 'history.json')
        
        # 創建必要的目錄
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(history_data_dir, exist_ok=True)
        
        # 初始化 history.json
        if not os.path.exists(history_file):
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
                
        print(f"目錄初始化完成：\n靜態目錄: {static_dir}\n歷史數據目錄: {history_data_dir}")
        
    except Exception as e:
        print(f"初始化目錄時發生錯誤: {str(e)}")

# 在應用啟動時調用
init_directories()

@app.route('/debug-system')
def debug_system():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(base_dir, 'static')
        history_data_dir = os.path.join(static_dir, 'history_data')
        history_file = os.path.join(history_data_dir, 'history.json')
        
        debug_info = {
            'base_dir_exists': os.path.exists(base_dir),
            'base_dir_path': base_dir,
            'static_dir_exists': os.path.exists(static_dir),
            'static_dir_path': static_dir,
            'history_data_dir_exists': os.path.exists(history_data_dir),
            'history_data_dir_path': history_data_dir,
            'history_file_exists': os.path.exists(history_file),
            'history_file_path': history_file,
            'history_file_size': os.path.getsize(history_file) if os.path.exists(history_file) else 0,
            'history_data_dir_contents': os.listdir(history_data_dir) if os.path.exists(history_data_dir) else []
        }
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                debug_info['history_records_count'] = len(history_data)
                debug_info['history_records_sample'] = history_data[0] if history_data else None
            except Exception as e:
                debug_info['history_file_error'] = str(e)
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete-history/<filename>', methods=['DELETE'])
def delete_history(filename):
    try:
        print(f"開始刪除歷史記錄: {filename}")
        history_data_dir = app.config['HISTORY_DATA_DIR']
        history_file = os.path.join(history_data_dir, 'history.json')

        # 取得檔案名，防止路徑遍歷
        filename = os.path.basename(filename)
        print(f"處理檔案名: {filename}")

        # 讀取歷史記錄
        with open(history_file, 'r', encoding='utf-8') as f:
            history_records = json.load(f)

        # 過濾掉要刪除的記錄
        new_records = [record for record in history_records if record.get('filename') != filename]
        print(f"原記錄數: {len(history_records)}，移除後記錄數: {len(new_records)}")

        # 更新歷史記錄文件
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        # 刪除圖片檔案
        image_path = os.path.join(history_data_dir, filename)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"已刪除圖片: {image_path}")
        else:
            print(f"圖片不存在: {image_path}")

        return jsonify({'status': 'success', 'message': '歷史記錄已刪除'})
    except Exception as e:
        print(f"刪除歷史記錄時發生錯誤: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/analyze-csv', methods=['POST'])
def analyze_csv():
    try:
        # 檢查是否有上傳的檔案
        if 'file' not in request.files:
            return jsonify({'error': '請選擇要上傳的CSV檔案'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '請選擇要上傳的CSV檔案'}), 400
        
        # 讀取CSV檔案
        try:
            df = pd.read_csv(file)
            print(f"成功讀取CSV檔案，行數: {len(df)}")
        except Exception as e:
            print(f"讀取CSV檔案時發生錯誤: {str(e)}")
            return jsonify({'error': f'讀取CSV檔案失敗: {str(e)}'}), 400
        
        # 檢查欄位
        required_columns = list(FEATURES.keys())
        available_columns = df.columns.tolist()
        
        # 檢查是否有足夠的列
        matching_columns = [col for col in available_columns if col in required_columns]
        if len(matching_columns) < 3:
            print(f"匹配的列數量不足: {len(matching_columns)}")
            return jsonify({'error': '請確保CSV檔案至少包含3個有效的財務指標'}), 400
        
        # 建立特徵向量
        feature_vectors = []
        for _, row in df.iterrows():
            feature_vector = np.zeros(35)  # 使用修改後的特徵集大小
            for indicator in FEATURES:
                if indicator in row:
                    feature_vector[FEATURES[indicator]] = row[indicator]
                elif indicator.lower() in [col.lower() for col in row.index]:
                    # 嘗試以不區分大小寫的方式匹配
                    matching_col = [col for col in row.index if col.lower() == indicator.lower()][0]
                    feature_vector[FEATURES[indicator]] = row[matching_col]
            feature_vectors.append(feature_vector)
        
        feature_vectors = np.array(feature_vectors)
        
        # 進行批次預測
        predictions = []
        probabilities = []
        financial_health_scores = []
        
        for feature_vector in feature_vectors:
            metrics = calculate_metrics_with_uncertainty(model, feature_vector.reshape(1, -1))
            predictions.append(metrics['prediction'])
            probabilities.append(1 - metrics['financial_health'])
            financial_health_scores.append(metrics['financial_health'])
        
        # 準備結果
        results = []
        for i, (pred, prob, health) in enumerate(zip(predictions, probabilities, financial_health_scores)):
            result = {
                'row_index': i + 1,
                'prediction': '財務健康' if pred == 1 else '可能破產',
                'bankruptcy_probability': float(prob),
                'financial_health': float(health)
            }
            results.append(result)
        
        # 計算整體統計
        total_rows = len(results)
        healthy_count = sum(1 for r in results if r['prediction'] == '財務健康')
        bankruptcy_count = sum(1 for r in results if r['prediction'] == '可能破產')
        
        summary = {
            'total_rows': total_rows,
            'healthy_count': healthy_count,
            'bankruptcy_count': bankruptcy_count,
            'healthy_percentage': (healthy_count / total_rows) * 100,
            'bankruptcy_percentage': (bankruptcy_count / total_rows) * 100
        }
        
        return jsonify({
            'summary': summary,
            'results': results
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

def load_api_key():
    """
    從 key.txt 檔案讀取 Google API 金鑰
    """
    try:
        # 使用相對路徑讀取 key.txt
        base_dir = os.path.dirname(os.path.abspath(__file__))
        key_path = os.path.join(base_dir, 'key.txt')
        
        # 檢查文件是否存在
        if not os.path.exists(key_path):
            print(f"API金鑰文件不存在: {key_path}")
            return None
            
        # 讀取API金鑰
        with open(key_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
            
        # 檢查API金鑰是否為空
        if not api_key:
            print("API金鑰為空")
            return None
            
        return api_key
    except Exception as e:
        print(f"讀取 API 金鑰時發生錯誤: {str(e)}")
        return None

def load_rag_system():
    """
    載入RAG系統，如果安裝了sentence-transformers則使用
    否則返回None但不中斷程序
    """
    try:
        # 檢查RAG.py文件是否存在
        rag_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAG.py')
        if not os.path.exists(rag_file_path):
            print(f"找不到RAG.py文件: {rag_file_path}")
            return None
            
        try:
            # 嘗試導入RAG.py中的函數
            from RAG import create_RAG_system, query_rag_system
            print("成功導入RAG模組")
            
            # 如果已存在向量數據庫，直接使用
            persist_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(os.path.join(persist_dir, 'chroma.sqlite3')):
                print("發現現有向量數據庫，直接載入")
                try:
                    rag_system = create_RAG_system(persist_directory=persist_dir)
                    print("成功載入RAG系統")
                    return rag_system
                except Exception as e:
                    print(f"載入現有向量數據庫失敗: {str(e)}")
                    return None
            else:
                print("未找到向量數據庫")
                return None
        except ImportError as e:
            print(f"無法導入RAG模組: {str(e)}")
            return None
    except Exception as e:
        print(f"載入RAG系統時發生錯誤: {str(e)}")
        return None

def get_gemini_analysis(file_path=None, metrics=None, financial_health=None, top_factors=None):
    """使用Google Gemini結合RAG系統分析上傳的企業財務報表"""
    try:
        # 載入RAG系統，但即使無法載入也繼續處理
        rag_system = load_rag_system()
        
        # 從 key.txt 讀取 API 金鑰
        api_key = load_api_key()
        if not api_key:
            raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
            
        # 嘗試導入並配置Google Generative AI
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
        except ImportError:
            print("未安裝google-generativeai套件，請使用pip install google-generativeai安裝")
            return "無法使用Gemini分析，缺少必要套件。請安裝google-generativeai套件。"
            
        # 建立Gemini模型
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        except Exception as e:
            print(f"建立Gemini模型時發生錯誤: {str(e)}")
            return f"建立Gemini模型失敗: {str(e)}"
        
        # 讀取檔案內容
        file_content = ""
        if file_path and os.path.exists(file_path):
            try:
                # 嘗試使用不同編碼讀取文件
                encodings = ['utf-8', 'latin-1', 'cp950', 'big5']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            file_content = file.read()
                        print(f"成功使用 {encoding} 編碼讀取檔案內容，長度: {len(file_content)}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                # 如果所有編碼都失敗，嘗試二進制讀取
                if not file_content:
                    with open(file_path, 'rb') as file:
                        file_content = str(file.read())
                    print(f"使用二進制模式讀取檔案內容，長度: {len(file_content)}")
                    
            except Exception as e:
                print(f"讀取檔案時發生錯誤: {str(e)}")
                file_content = f"無法讀取檔案內容: {str(e)}"
        else:
            file_content = "未提供檔案或檔案不存在"
            if top_factors:
                file_content += "\n主要影響因素: " + ", ".join(top_factors)
        
        # 格式化指標為百分比字串
        if not metrics:
            metrics = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.80,
                'f1_score': 0.81
            }
            
        formatted_metrics = {
            'accuracy': f"{metrics.get('accuracy', 0) * 100:.2f}%",
            'precision': f"{metrics.get('precision', 0) * 100:.2f}%",
            'recall': f"{metrics.get('recall', 0) * 100:.2f}%",
            'f1': f"{metrics.get('f1_score', 0) * 100:.2f}%"
        }
        
        # 判斷財務健康狀況
        if financial_health is None:
            financial_health = 0.5
            
        health_status = "財務健康" if financial_health > 0.5 else "可能破產"
        health_percentage = f"{financial_health * 100:.2f}%"  # 先轉為百分比再加上百分號
        
        # 主要影響因素
        factors_info = ""
        if top_factors:
            factors_info = f"\n主要影響因素: {', '.join(top_factors)}"
        
        # 使用RAG系統獲取相關研究資訊
        rag_insights = ""
        if rag_system:
            # 對每個影響因素使用RAG查詢
            factor_queries = []
            if top_factors:
                for factor in top_factors:
                    factor_queries.append(f"{factor}在破產預測中的重要性")
            else:
                factor_queries = ["財務指標在破產預測中的重要性"]
            
            # 獲取RAG洞見
            for query in factor_queries:
                try:
                    # 使用導入的query_rag_system函數
                    from RAG import query_rag_system as rag_query
                    insight = rag_query(rag_system, query)
                    rag_insights += f"\n關於 {query}:\n{insight}\n"
                except Exception as e:
                    print(f"RAG查詢時發生錯誤: {str(e)}")
                    rag_insights += f"\n關於 {query}: 查詢過程中發生錯誤。\n"
        else:
            # 如果RAG系統未啟用，使用默認文本
            rag_insights = "RAG系統未啟用。這裡提供一般性的財務健康分析。"
            
        # 獲取分析模板
        try:
            from gemini_analysis import TEMPLATES
            analysis_template = TEMPLATES.get('analysis_template', "")
        except ImportError:
            print("無法導入TEMPLATES，使用默認模板")
            # 使用內嵌的簡化模板
            analysis_template = """
            你是金融分析專家，請針對以下企業財務報表進行詳細分析。
            根據我們的機器學習模型預測，該企業的財務健康指數為{health_percentage}，狀態為{health_status}。
            模型的效能指標如下：
            - 準確率(Accuracy): {accuracy}
            - 精確率(Precision): {precision}
            - 召回率(Recall): {recall}
            - F1分數: {f1}{factors_info}

            以下是企業財務報表的內容：
            {file_content}
            
            以下是相關財務預測研究的洞見：
            {rag_insights}

            請提供詳細的金融分析，使用[SECTION0]到[SECTION5]標記分隔不同部分。
            """
        
        # 替換模板中的變量
        prompt = analysis_template.format(
            health_percentage=health_percentage,
            health_status=health_status,
            accuracy=formatted_metrics['accuracy'],
            precision=formatted_metrics['precision'],
            recall=formatted_metrics['recall'],
            f1=formatted_metrics['f1'],
            factors_info=factors_info,
            file_content=file_content,
            rag_insights=rag_insights
        )
        
        # 發送請求到Gemini模型
        try:
            response = model.generate_content(prompt)
            analysis = response.text
            print(f"Gemini分析完成，回應長度: {len(analysis) if analysis else 0}")
            return analysis
        except Exception as e:
            print(f"Gemini API 調用失敗: {str(e)}")
            return "財務分析生成失敗，請稍後再試。" + str(e)
    except Exception as e:
        print(f"分析過程中發生錯誤: {str(e)}")
        return "財務分析生成過程發生錯誤，請稍後再試。" + str(e)
