from bert_score import score
from bert_score import plot_example
from rouge import Rouge


def bert_score_compute(st_cands, st_ref, lang):
    cands = st_cands.split(".")
    refs = st_ref.split(".")
    P, R, F1 = score(cands, refs, lang=lang,
                     model_type="bert-base-multilingual-cased", verbose=True)
    return round(float(P.mean()), 2), float(R.mean()), round(float(F1.mean()), 3)


def plot_similarity_matrix(cand, ref, lang):
    plot_example(cand, ref, lang=lang)


def rouge_score_compute(cands, refs, rouge_type):
    rouge = Rouge()
    scores = rouge.get_scores(cands, refs)[0]
    P = scores["rouge-" + rouge_type]['p']
    F1 = scores["rouge-" + rouge_type]['f']
    R = scores["rouge-" + rouge_type]['r']
    return round(P, 3), round(R, 3), round(F1, 3)


if __name__ == '__main__':
    # cands = "tôi rất đẹp trai. Tôi đi nhiều nơi trên thế giới"
    # refs = "tôi đẹp trai và đã đi nhiều nơi trên thế giới"
    cands = "Bộ Khoa học và Công nghệ tổ chức Hội nghị công bố Quyết định kiểm toán. Tham dự Hội nghị có Thứ trưởng Bộ KH&CN Trần Việt Thanh, Vụ trưởng Vụ Tài chính Nguyễn Ngọc Song và đại diện lãnh đạo các đơn vị thuộc Bộ có liên quan. Được biết, các đại biểu tham dự đã có sự thống nhất về việc điều chỉnh thời gian sao cho phù hợp với từng điều kiện cụ thể tại mỗi đơn vị được kiểm toán để có sự phối hợp tốt nhất đối với công việc của các bên liên quan, thời gian kiểm toán sẽ bắt đầu từ ngày 09/7 và kết thúc vào ngày 10/9/2013."
    refs = "Bộ Khoa học và Công nghệ đã tổ chức Hội nghị công bố Quyết định kiểm toán.Mục tiêu kiểm toán nhằm xác định tính đúng đắn, trung thực hợp lý của Báo cáo quyết toán ngân sách năm 2012 của Bộ KH&CN; đánh giá tính kinh tế, hiệu lực và hiệu quả trong quản lý, sử dụng ngân sách, tiền, tài sản nhà nước tại các đơn vị được kiểm toán.Phạm vi kiểm toán là báo cáo quyết toán ngân sách năm 2012 của Bộ KH&CN, báo cáo quyết toán năm 2011, 2012 của Quỹ phát triển KH&CN và các thời kỳ trước, sau có liên quan của các đơn vị được kiểm toán.Vụ Tài chính sẽ là đầu mối thông tin cho các đơn vị thuộc Bộ và các đơn vị có liên quan.Thời gian kiểm toán sẽ bắt đầu từ ngày 09/7 và kết thúc vào ngày 10/9/2013."
    # P, R, F1 = rouge_score_compute(cands, refs, 'l')
    P, R, F1 = rouge_score_compute(cands, refs, 'l')
    print(P)
    print(R)
    print(F1)
