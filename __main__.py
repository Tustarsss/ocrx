import re
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz
from docx import Document
import os

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

def load_document(docx_path):
    doc = Document(docx_path)
    filtered = [
        para.text.strip() 
        for para in doc.paragraphs 
        if para.text.strip() and not para.text.startswith("-----------------------")
    ]
    full_text = ' '.join(filtered)
    return preprocess_text(full_text)

def sliding_window_match(ocr_line, doc_text, threshold=100):
    ocr_len = len(ocr_line)
    window_size = ocr_len
    max_start = len(doc_text) - window_size
    best_score = 0
    best_match = ""
    
    if max_start < 0:
        window_size = len(doc_text)
        max_start = 0
    
    for i in range(0, max_start + 1):
        window = doc_text[i:i+window_size]
        score = fuzz.ratio(ocr_line, window)
        if score > best_score:
            best_score = score
            best_match = window
        if score >= threshold:
            return (True, best_match, best_score)
    return (best_score >= threshold, best_match, best_score)

def verify_text_match(image_path, docx_path):
    ocr = PaddleOCR(
        use_angle_cls=True,
        det_model_dir='./ocr/PP-OCRv4_mobile_det_infer',
        rec_model_dir='./ocr/PP-OCRv4_mobile_rec_infer',
        use_gpu=False,
        det_limit_side_len=2048,
        drop_score=0.3
    )
    
    result = ocr.ocr(image_path, cls=True)
    ocr_raw = [line[1][0] for line in result[0]]
    ocr_processed = [correct_ocr_errors(preprocess_text(line)) for line in ocr_raw]
    doc_text = load_document(docx_path)
    print(doc_text)
    unmatched = []
    for raw_line, proc_line in zip(ocr_raw, ocr_processed):
        if len(proc_line) < 2:
            continue
        matched, best_match, score = sliding_window_match(proc_line, doc_text)
        if not matched:
            unmatched.append((raw_line, best_match, score))
    
    return unmatched

def process_docs_and_images(doc_dir='./docx', pic_dir='./picture', output_file='./unmatched_results.txt'):
    doc_filenames = set([os.path.splitext(f)[0] for f in os.listdir(doc_dir) if f.endswith('.docx')])
    image_filenames = set([os.path.splitext(f)[0] for f in os.listdir(pic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    
    matched_pairs = doc_filenames.intersection(image_filenames)
    results = []
    
    for filename in matched_pairs:
        doc_file = os.path.join(doc_dir, filename + '.docx')
        image_file = os.path.join(pic_dir, filename + '.jpg')
        unmatched = verify_text_match(image_file, doc_file)
        if unmatched:
            results.append((filename, unmatched))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if results:
            for filename, entries in results:
                f.write(f"文件: {filename}\n")
                for raw_line, best_match, score in entries:
                    f.write(f"\n- OCR未匹配内容: {raw_line}\n")
                    if score>50:
                        f.write(f"  最可能原文:\t{best_match} (相似度: {score}%)\n")
                print("\n")
        else:
            f.write("所有文字均匹配对应文档内容\n")

if __name__ == '__main__':
    process_docs_and_images()
    print("处理完成，结果已保存至当前目录下的'unmatched_results.txt'文件中。")
    os.system("pause")