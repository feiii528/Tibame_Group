import google.generativeai as genai
import os
import numpy as np
import chardet
def load_api_key():
    """
    從 key.txt 檔案讀取 Google API 金鑰
    """
    try:
        # 使用相對路徑讀取 key.txt
        key_path = os.path.join('C:/Users/ASUS/Desktop/online/專題/團體/Bankrupt/flask_project/key.txt')
        with open(key_path, 'r') as f:
            api_key = f.read().strip()
        return api_key
    except Exception as e:
        print(f"讀取 API 金鑰時發生錯誤: {str(e)}")
        return None

def get_gemini_analysis(file_path=None, metrics=None, financial_health=None, top_factors=None):
    """使用Google Gemini分析上傳的企業財務報表"""
    try:
        # 從 key.txt 讀取 API 金鑰
        api_key = load_api_key()
        if not api_key:
            raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
        
        genai.configure(api_key=api_key)
        
        # 建立Gemini模型
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        # 讀取檔案內容
        file_content = ""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected_encoding = chardet.detect(raw_data)['encoding']
                with open(file_path, 'r', encoding=detected_encoding, errors='replace') as file:
                    file_content = file.read()
                print(f"成功讀取檔案內容，長度: {len(file_content)}")
            except Exception as e:
                print(f"讀取檔案時發生錯誤: {str(e)}")
                file_content = "無法讀取檔案內容"
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
        health_percentage = f"{financial_health * 100:.2f}%"
        
        # 主要影響因素
        factors_info = ""
        if len(top_factors) > 0:
            factors_info = f"\n主要影響因素: {', '.join(top_factors)}"
        
        # 建立提示詞
        prompt = f"""
        你是金融分析專家，請針對以下企業財務報表進行詳細分析。
        根據我們的機器學習模型預測，該企業的財務健康度為{health_percentage}，狀態為{health_status}。
        模型的效能指標如下：
        - 準確率(Accuracy): {formatted_metrics['accuracy']}
        - 精確率(Precision): {formatted_metrics['precision']}
        - 召回率(Recall): {formatted_metrics['recall']}
        - F1分數: {formatted_metrics['f1']}
        - 影響前3大因素: {factors_info}

        以下是企業財務報表的內容：
        {file_content}

        請提供詳細的金融分析，針對該企業財務狀況給出專業見解和改進建議。

        ***極其重要的格式要求***：
        你的回答必須嚴格按照以下格式，分為六個明確標記的部分：

        [SECTION0]
        模型資訊與整體評估：簡要說明使用的模型，提供該企業財務健康的整體評估。財務健康度和模型效能指標請不要重複加上百分號，例如財務健康度49.67% 應寫作49.67%，而不是49.67%%；準確率50%應寫作0.5235，而不是0.50%。

        [SECTION1]
        整體統計分析：針對主要財務指標進行統計分析，指出關鍵數據點的意義。

        [SECTION2]
        詳細指標解釋：深入解釋各項財務指標的含義及其對企業營運的影響。

        [SECTION3]
        改進建議（低財務健康時）：如果財務健康較低，提供具體可行的改進建議；或如果財務健康良好，提供維持與進一步提升的建議。

        [SECTION4]
        建議深入分析方向：指出可能需要進一步分析的領域或財務方面。

        [SECTION5]
        注意事項：請注意，以上建議為AI隨機生成僅供參考，企業應根據自身具體情況，審慎評估並制定相應的策略！

        請注意：
        1. 必須嚴格遵循上述格式，每個部分必須以[SECTIONx]開頭（x為0-5的數字）
        2. 不要添加其他標題或分隔符
        3. 每個部分的內容必須專業、詳細且有針對性
        4. 請直接以[SECTION0]開始你的回答，不要有任何前導文字
        5. 每個部分獨立完整，不要在不同部分間相互引用
        6. 分析必須基於提供的財務報表內容、模型指標和研究洞見
        7. 遇到數字開頭的段落時，請自動換行
        8. 請確保財務健康度和各項指標的百分比格式正確，不要重複添加百分號
        9. 每一句話後請自動換行，每個段落之間保留空行
        10. 使用繁體中文回覆訊息
        """
        
        # 發送請求到Gemini API
        response = model.generate_content(prompt)
        
        # 檢查回應是否成功
        if response.text and len(response.text) > 100:
            return response.text
        else:
            print("Gemini API 返回內容過短")
            return "Gemini API 返回內容過短或為空，請稍後再試。"
            
    except Exception as e:
        print(f"使用Gemini分析時發生錯誤: {e}")
        return f"無法進行Gemini分析，錯誤: {str(e)}"

def analyze_batch_results(metrics=None, top_factors=None):
    """
    分析批次預測結果
    
    Parameters:
    results (list): 批次預測結果列表
    
    Returns:
    str: Gemini 的分析結果
    """
    # 從 key.txt 讀取 API 金鑰
    api_key = load_api_key()
    if not api_key:
        raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
    
    genai.configure(api_key=api_key)

    # 主要影響因素
    factors_info = ""
    if len(top_factors) > 0:
        factors_info = f"{', '.join(top_factors)}"
    # 計算整體統計
    # total_rows = len(metrics)
    # healthy_count = sum(1 for r in metrics if isinstance(r, dict) and r.get('prediction') == 0)
    # bankruptcy_count = sum(1 for r in metrics if isinstance(r, dict) and r.get('prediction') == 1)
    health_status = "財務健康" if metrics['financial_health'] > 50 else "可能破產"
    health_percentage = f"{metrics['financial_health'] :.2f}%"
    

    # 計算平均指標
    avg_metrics = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }
    #[UNDO]
    # accuracy = [0.7 + 0.2 * r['accuracy'] for r in metrics if isinstance(r, dict) and 'financial_health' in r]
    # precision = [0.65 + 0.2 * r['precision'] for r in metrics if isinstance(r, dict) and 'financial_health' in r]
    # recall = [0.6 + 0.2 * r['recall'] for r in metrics if isinstance(r, dict) and 'financial_health' in r]
    # f1 = [0.625 + 0.2 * r['f1'] for r in metrics if isinstance(r, dict) and 'financial_health' in r]
    # 計算平均指標
    # avg_metrics = {
    #     'accuracy': np.mean(accuracy),
    #     'precision': np.mean(precision),
    #     'recall': np.mean(recall),
    #     'f1': np.mean(f1)
    # }
    #   整體統計：
    #       總樣本數：{total_rows}
    #       財務健康企業數：{healthy_count} ({healthy_count/total_rows*100:.2f}%)
    #       可能破產企業數：{bankruptcy_count} ({bankruptcy_count/total_rows*100:.2f}%)  
    #       準備提示詞
    #[UNDO]
    prompt = f"""
    你是專業的金融分析師，以精簡且專業的答覆回應客戶，分析以下批次評估結果並提供個性化的改善建議：

    平均評估指標：
    預測結果：{health_status}
    財務健康度：{health_percentage}
    準確率：{round(np.mean(avg_metrics['accuracy'])*100, 2)}
    精確率：{round(np.mean(avg_metrics['precision'])*100, 2)}
    召回率：{round(np.mean(avg_metrics['recall'])*100, 2)}
    F1分數：{round(np.mean(avg_metrics['f1'])*100, 2)}
    影響前3大因素：{factors_info}
    

    必須嚴格按照以下格式回答，你的回覆必須包含完整的6個部分，每個部分都以[SECTION數字]開頭：

    [SECTION0]
    模型資訊
    (簡要説明模型分析結果和批次評估情況，財務健康度和模型效能指標請不要重複加上百分號，例如財務健康度49.67%應寫作49.67，而不是49.67%%；準確率52.35%應寫作0.5235，而不是0.50%。)

    [SECTION1]
    整體統計的數據與指標分析
    (分析企業樣本分布和整體財務健康狀況的洞察)

    [SECTION2]
    對這些指標的詳細解釋
    (解釋批次預測指標對企業群體財務狀況的實際意義)

    [SECTION3]
    針對可能破產企業的具體改善建議
    (提供針對性的、切實可行的改善策略)

    [SECTION4]
    建議的後續分析方向
    (提供更深入群組分析的方向和建議)

    [SECTION5]
    注意事項：請注意，以上建議為AI隨機生成僅供參考，企業應根據自身具體情況，審慎評估並制定相應的策略！

    嚴格遵循以下規則：
    請注意：
    1. 必須嚴格遵循上述格式，每個部分必須以[SECTIONx]開頭（x為0-5的數字）
    2. 不要添加其他標題或分隔符
    3. 每個部分的內容必須專業、詳細且有針對性
    4. 請直接以[SECTION0]開始你的回答，不要有任何前導文字
    5. 每個部分獨立完整，不要在不同部分間相互引用
    6. 分析必須基於提供的財務報表內容、模型指標和研究洞見
    7. 遇到數字開頭的段落時，請自動換行
    8. 請確保財務健康度和各項指標的百分比格式正確，不要重複添加百分號
    9. 每一句話後請自動換行，每個段落之間保留空行
    10. 使用繁體中文回覆訊息
    """
    
    # 使用 Gemini 進行分析
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    
    return response.text 
