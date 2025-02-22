import re
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz
from docx import Document
import os
from PIL import Image, ImageDraw
from typing import List, Dict

def correct_ocr_errors(text):
    error_map = {
        r"照[贵櫃]": "昭贵",
        r"股贵": "昭贵",
        r"zhaxgui": "zhaogui",
    }
    for pattern, replacement in error_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', '', text)
    return text

def load_all_documents(doc_dir: str) -> Dict[str, str]:
    """加载所有文档内容并返回{文件名: 预处理文本}"""
    docs = {}
    for filename in os.listdir(doc_dir):
        if not filename.endswith('.docx'):
            continue
        path = os.path.join(doc_dir, filename)
        doc = Document(path)
        filtered = [
            para.text.strip() 
            for para in doc.paragraphs 
            if para.text.strip() and not para.text.startswith("-----------------------")
        ]
        full_text = ' '.join(filtered)
        docs[filename] = preprocess_text(full_text)
    return docs

def text_exists_in_any_doc(proc_text: str, all_docs: Dict[str, str], threshold=100) -> bool:
    """检查文本是否存在于任意文档中"""
    if len(proc_text) < 2:
        return True  # 忽略短文本
    
    for doc_text in all_docs.values():
        if sliding_window_match(proc_text, doc_text, threshold)[0]:
            return True
    return False

def sliding_window_match(ocr_line: str, doc_text: str, threshold):
    ocr_len = len(ocr_line)
    window_size = ocr_len
    max_start = len(doc_text) - window_size
    best_score = 0
    
    if max_start < 0:
        return (False, "", 0)
    
    for i in range(0, max_start + 1):
        window = doc_text[i:i+window_size]
        score = fuzz.ratio(ocr_line, window)
        if score > best_score:
            best_score = score
        if score >= threshold:
            return (True, window, score)
    return (best_score >= threshold, "", best_score)

def process_check_folder(check_dir='./核对内容', output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 验证文件
    docs = [f for f in os.listdir(check_dir) if f.endswith('.docx')]
    images = [f for f in os.listdir(check_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if len(images) != 1:
        raise ValueError("核对文件夹中应包含且仅包含一个图片文件")
    if not docs:
        raise ValueError("核对文件夹中至少需要一个文档")

    # 初始化OCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        det_model_dir='./ocr/PP-OCRv4_mobile_det_infer',
        rec_model_dir='./ocr/PP-OCRv4_mobile_rec_infer',
        use_gpu=False,
        det_limit_side_len=2048,
        drop_score=0.3
    )

    # 加载数据
    image_path = os.path.join(check_dir, images[0])
    all_docs = load_all_documents(check_dir)
    
    # OCR识别
    result = ocr.ocr(image_path, cls=True)
    ocr_data = [(line[1][0], [tuple(map(int, p)) for p in line[0]]) for line in result[0]]
    
    # 匹配检测
    unmatched = []
    for text, coords in ocr_data:
        proc_text = correct_ocr_errors(preprocess_text(text))
        if not text_exists_in_any_doc(proc_text, all_docs):
            unmatched.append((text, coords))

    # 生成结果
    if unmatched:
        # 绘制标记图片
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for _, coords in unmatched:
            draw.polygon(coords, outline="red", width=3)
        
        marked_path = os.path.join(output_dir, f"marked_{images[0]}")
        image.save(marked_path)
        
        # 生成报告
        report_path = os.path.join(output_dir, "未匹配报告.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"图片文件: {images[0]}\n")
            f.write(f"涉及文档: {', '.join(docs)}\n\n")
            f.write("未匹配文本区域:\n")
            for text, _ in unmatched:
                f.write(f"- {text}\n")
        
        print(f"发现{len(unmatched)}处未匹配内容，已标记在 {marked_path}")
        os.system(f"start{marked_path}")
    else:
        print("所有文本内容均在文档中找到匹配")

if __name__ == '__main__':
    process_check_folder()
    os.system("pause")