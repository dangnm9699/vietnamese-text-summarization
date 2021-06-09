from bert_score import score
from bert_score import plot_example
from rouge import Rouge


def bert_score_compute(st_cands, st_ref, lang):
    cands = st_cands.split(".")
    refs = st_ref.split(".")
    P, R, F1 = score(cands, refs, lang=lang, model_type="bert-base-multilingual-cased", verbose=True)
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
    cands = "Đại tướng Lê Đức Anh nhận huy hiệu 75 năm tuổi Đảng. Thay mặt lãnh đạo Đảng, Nhà nước, Quân ủy Trung ương, Tổng Bí thư Nguyễn Phú Trọng đã trân trọng trao Huy hiệu 75 năm tuổi Đảng và nhiệt liệt chúc mừng đồng chí Lê Đức Anh được nhận phần thưởng cao quý của Đảng và Nhà nước. Với tác phong sâu sát cơ sở, giữ vững nguyên tắc, tôn trọng và lắng nghe ý kiến cấp dưới, đồng chí luôn giữ được mối quan hệ đoàn kết gắn bó, phát huy được trí tuệ tập thể trong lãnh đạo, chỉ huy. Đồng chí Lê Đức Anh nêu rõ: Là một đảng viên dưới sự lãnh đạo của Đảng, đồng chí đã hoàn thành nhiều nhiệm vụ Đảng giao, vì dân, vì nước, vì bạn bè quốc tế. Đồng chí mong mỏi và tuyệt đối tin tưởng, dưới sự lãnh đạo của Đảng, toàn dân, toàn quân ta sẽ giữ vững độc lập dân tộc, chủ quyền quốc gia, thực hiện các chủ trương, chính sách nhất quán của Đảng và Nhà nước, vì mục tiêu hòa bình, hữu nghị, hợp tác cùng phát triển với các nước trên thế giới, đưa nước ta vững bước tiến lên Chủ nghĩa Xã hội, xây dựng đất nước Việt Nam giàu mạnh, dân chủ, công bằng, văn minh."
    refs = "Chiều 29/7, Đại tướng Lê Đức Anh, nguyên Ủy viên Bộ Chính trị, nguyên Phó Bí thư Quân ủy Trung ương - Bộ trưởng Bộ Quốc phòng, nguyên Chủ tịch nước CHXHCN Việt Nam, nguyên Cố vấn Ban Chấp hành Trung ương Đảng đã nhận huy hiệu 75 năm tuổi Đảng. Đồng chí Lê Đức Anh là một cán bộ đã từng kinh qua nhiều cương vị công tác, được tôi luyện, trưởng thành qua các cuộc kháng chiến, có nhiều đóng góp to lớn cho sự nghiệp cách mạng và sự trưởng thành của Quân đội.  Chủ tịch nước Trương Tấn Sang, đồng chí lãnh đạo, nguyên lãnh đạo Đảng, Nhà nước, đã trao tặng những bó hoa  chúc mừng. Đồng chí Lê Đức Anh đã bày tỏ lòng biết ơn sâu sắc, niềm vinh dự to lớn được đón nhận Huy hiệu 75 năm tuổi Đảng, phần thưởng cao quý của Đảng và Nhà nước."
    # P, R, F1 = rouge_score_compute(cands, refs, 'l')
    P, R, F1 = rouge_score_compute(cands, refs, '1')
    print(P)
    print(R)
    print(F1)
