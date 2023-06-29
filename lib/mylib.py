import streamlit as st
import pickle
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize
from tensorflow.keras.preprocessing import sequence
import numpy as np
from tensorflow.keras.models import load_model
import base64
import pandas as pd



# Ìnormation app
name_app = 'Sentiment Analysis'
version_app = '1.0'
current_time = '2023-06-29 11:16:58'
model_setiment_analysis = None
vectorizer = None
file_name_css = 'assets/css/styles.css'


# Load file
##LOAD EMOJICON
file = open('Data/files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()

#LOAD TEENCODE
file = open('Data/files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()


## LOAD STOPWORDS
file = open('Data/files/vietnamese_stopword_food.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

with open('Data/files/special_phrases.txt', encoding='utf-8') as file:
    special_phrases = file.read().splitlines()

report_df = pd.read_csv('Data/report.csv')


max_words = 5400
max_len = 75


# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def add_underscore(text):
    
    for phrase in special_phrases:
        text = text.replace(phrase, phrase.replace(' ', '_'))
    
    return text

def process_special_word(text):
    text = add_underscore(text)
    new_text = ''
    text_lst = text.split()
    i = 0
    special_words = ['không', 'chưa', 'không_hề', 'không_thể', 'chẳng_có', 'chẳng_hề', 'chẳng_thể', 'không_có', 'không_phải', 'chả', 'không_còn', 'kém', 'không_bị', 'không_quá', 'không_được']
    
    if any(word in text_lst for word in special_words):
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            
            if word in special_words:
                next_idx = i + 1
                if next_idx <= len(text_lst) - 1:
                    word = word + '_' + text_lst[next_idx]
                i = next_idx + 1
            else:
                i = i + 1
            
            new_text = new_text + word + ' '
    else:
        new_text = text
    
    return new_text.strip()


def remove_stopword(text, stopwords):
    document = ' '.join('' if word in stopwords else word for word in text.split())
    
    # DEL excess blank space
    document = re.sub(r'\s+', ' ', document).strip()
    return document


def load_model_setiment_analysis():
    global model_setiment_analysis
    
    # Nếu model đã được load trước đó thì không cần load lại
    if model_setiment_analysis is not None:
        return model_setiment_analysis
    
    # Nếu model chưa được load thì load model lưu vào biến toàn cục
    model_setiment_analysis = load_model('models/model.h5')
    
    return model_setiment_analysis

def load_vectorizer():
    global vectorizer
    
    # Nếu model đã được load trước đó thì không cần load lại
    if vectorizer is not None:
        return vectorizer
    
    # Nếu model chưa được load thì load model từ file pkl và lưu vào biến toàn cục
    pkl_filename = 'models/tokenizer.pkl'
    with open(pkl_filename, 'rb') as file:  
        vectorizer = pickle.load(file)
    
    return vectorizer


def Introduce():
    st.subheader('Business Objective')
    st.write("""**Vấn đề hiền tại:** Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn *(Small/Large/XLarge Bags)*, và có 3 PLU (Product Look Up) khác nhau *(4046, 4225, 4770)*. Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng""")
    st.write("""
    **Mục tiêu/ Vấn đề:**
    => Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ => xem xét việc mở rộng sản xuất, kinh doanh.
    """)

def clean_text(text, emoji_dict, teen_dict):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    
    new_sentence =''
    for sentence in sent_tokenize(text):
        ###### CONVERT EMOJICON
        sentence = ''.join(' ' + emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(re.findall(pattern,sentence))

        new_sentence = new_sentence + sentence + '. '
    
    text = new_sentence
    text = re.sub(r'(\w)\1+', r'\1', text)
    text = text.replace("’",'')
    text = re.sub(r'\.+', ".", text)
    text = re.sub(r'\.+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\s*,\s*", ", ", text)

    return text


def predict_class(text, model):
    
    # Chuẩn bị dữ liệu đầu vào
    text = covert_unicode(text)
    text = clean_text(text, emoji_dict, teen_dict)
    text = process_special_word(text)
    text = word_tokenize(text, format="text")
    text = remove_stopword(text, stopwords_lst)
    text_sequence = vectorizer.texts_to_sequences([text])
    text_sequence_matrix = sequence.pad_sequences(text_sequence, maxlen=max_len)

    # Dự đoán lớp
    prediction = model.predict(text_sequence_matrix)[0]
    predicted_class = (prediction > 0.5).astype(int)

    # Chuyển đổi kết quả thành tên lớp
    class_names = ['Positive', 'Negative']
    predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name



def create_contact_form():
    # contact form
    with st.sidebar:
        st.write('Send us your feedback!')
        name_input = st.text_input('Your name')
        comment_input = st.text_area('Your comment')
        submitted = st.button('Submit')

        # Nếu người dùng đã gửi đánh giá, thêm đánh giá vào dataframe
        if submitted:
            # Thêm đánh đánh giá người dùng vào file txt
            pass

def create_infomation_app(name_app, version_app, current_time):
    # Infomations app
    st.sidebar.markdown(
        """
        <div style='position: fixed; bottom: 0'>
            <p> """+ name_app +""" - Version: """+ version_app +""" </br>(For Sentiment analysis of text)</p>
            <p><i>Last Updated: """+ current_time +"""<i/></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    


def show_introduction():
    st.sidebar.markdown("<h1 style='color: #2196f3;'>Wellcome to Sentiment analysis app!</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<p>Ứng dụng đơn giản để phân loại cảm xúc Like (positive) <span>|</span> Dislike (negative) dựa trên một comment của người dùng.</p>", unsafe_allow_html=True)


# def show_Objective():
#     st.subheader('Business Objective/Problem')
    
#     business_objective = """
#     - Foody.vn là một kênh phối hợp với các nhà hàng/quán ăn bán thực phẩm online.
#     - Chúng ta có thể lên đây để xem các đánh giá, nhận xét cũng như đặt mua thực phẩm.
#     - Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/sản phẩm.
#     - Xây dựng hệ thống dựa trên lịch sử những đánh giá của
#     khách hàng đã có trước đó. Dữ liệu được thu thập từ
#     phần bình luận và đánh giá của khách hàng ở
#     Foody.vn…

#     => **Mục tiêu/ vấn đề:** Xây dựng mô hình dự đoán giúp
#     nhà hàng có thể biết được những phản hồi nhanh chóng
#     của khách hàng về sản phẩm hay dịch vụ của họ (tích
#     cực, tiêu cực), điều này giúp cho nhà hàng
#     hiểu được tình hình kinh doanh, hiểu được ý kiến của
#     khách hàng từ đó giúp nhà hàng cải thiện hơn trong dịch
#     vụ, sản phẩm.
#         """
#     st.markdown(business_objective, unsafe_allow_html=True)

#     st.subheader('Data Understanding')
#     data_Understanding = """
#     Cung cấp dữ liệu data_Foody.csv có 39.925 mẫu gồm 3
#     thông tin: 
#     - Tên nhà hàng
#     - Nội dung review
#     - Điểm đánh giá

#     Có thể tập trung giải quyết bài toán Sentiment analysis in Cuisine Area với:
#     ▪ RNN, LSTM
#     ▪ PhoBERT
#         """
#     st.markdown(data_Understanding, unsafe_allow_html=True)
    
#     st.image('Data/1.png')
#     st.image('Data/2.png')
#     st.image('Data/3.png')

#     st.subheader('LSTM-based Model for Word Segmentation')
#     st.image('Data/4.png')
#     st.dataframe(report_df)


def show_Objective():
    st.subheader('Business Objective/Problem')
    
    business_objective = """
    - Foody.vn là một kênh phối hợp với các nhà hàng/quán ăn bán thực phẩm online.
    - Chúng ta có thể lên đây để xem các đánh giá, nhận xét cũng như đặt mua thực phẩm.
    - Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/sản phẩm.
    - Xây dựng hệ thống dựa trên lịch sử những đánh giá của
    khách hàng đã có trước đó. Dữ liệu được thu thập từ
    phần bình luận và đánh giá của khách hàng ở
    Foody.vn…

    => **Mục tiêu/ vấn đề:** Xây dựng mô hình dự đoán giúp
    nhà hàng có thể biết được những phản hồi nhanh chóng
    của khách hàng về sản phẩm hay dịch vụ của họ (tích
    cực, tiêu cực), điều này giúp cho nhà hàng
    hiểu được tình hình kinh doanh, hiểu được ý kiến của
    khách hàng từ đó giúp nhà hàng cải thiện hơn trong dịch
    vụ, sản phẩm.
        """
    st.markdown(business_objective, unsafe_allow_html=True)

    st.subheader('Data Understanding')
    data_Understanding = """
    Cung cấp dữ liệu data_Foody.csv có 39.925 mẫu gồm 3
    thông tin: 
    - Tên nhà hàng
    - Nội dung review
    - Điểm đánh giá

    Có thể tập trung giải quyết bài toán Sentiment analysis in Cuisine Area với:
    ▪ RNN, LSTM
    ▪ PhoBERT
        """
    st.markdown(data_Understanding, unsafe_allow_html=True)
    
    st.image('Data/1.png')
    st.image('Data/2.png')
    st.image('Data/3.png')

    st.subheader('LSTM-based Model for Word Segmentation')
    st.image('Data/4.png')
    st.dataframe(report_df)
    st.image('Data/5.png')
    
    st.subheader('Additional Information')
    additional_info = """
    - Trong quá trình xây dựng mô hình dự đoán, chúng ta sử dụng mạng LSTM để thực hiện việc phân đoạn từ.
    - Mô hình sử dụng kiến trúc RNN với một tầng LSTM và một tầng fully connected để dự đoán kết quả.
    - Các đặc trưng của từ được trích xuất thông qua việc sử dụng phương pháp word embedding.
    - Bảng dữ liệu báo cáo (report_df) chứa kết quả đánh giá của mô hình trên tập dữ liệu test.
    """
    st.markdown(additional_info, unsafe_allow_html=True)
