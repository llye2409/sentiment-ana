from lib.mylib import *
import streamlit as st

# Load model
vectorizer = load_vectorizer()
model_setiment_analysis = load_model_setiment_analysis()

# Page setting
st.set_page_config(page_title="Sentiment_analysis", page_icon=":😊:", layout="centered")

# Sidebar
show_introduction()
menu_selection = st.sidebar.radio("Menu", ["Introduction", "Prediction"])

# Main content
st.title("Sentiment Analysis App")

# Add your main content here
if menu_selection == "Introduction":
    st.write("This is the Introduction section.")
    show_Objective()

elif menu_selection == "Prediction":
    # # Add a file uploader to allow users to upload a txt file
    uploaded_file = st.file_uploader("Upload a txt file", type=["csv"])

    # Add a text area for users to input text
    labels = """
    Nhập văn bản (Ví dụ: ngon quá!!! | đồ ăn dở tệ | Quán ngon mà rẻ, phục vụ tốt 👍 | Trời mưa gọi về nhâm nhi đúng bài luôn!
    | Gà vừa ăn, 1 phần gà size s đủ cho 2-3 người ăn, view đẹp, nhân viên phục vụ tốt 
    Lần sau sẽ quay lại
    """

    # Add the "Predict" button for file upload
    if uploaded_file is not None:
        if st.button("Analyze", key=1):
            with st.spinner("Loading..."):
                df = pd.read_csv(uploaded_file, header=None)
                df[1] = df[0].apply(lambda x: predict_class(x, model_setiment_analysis))
                st.table(df.head())
            st.success("Prediction completed!")

            # Show download link for the CSV file
            data = df.to_csv(index=False, encoding="utf-8")
            b64 = base64.b64encode(data.encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="data.csv">Download data as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    # Add the "Predict" button for text input
    st.write("Predict from Text Input")
    st.write(labels)
    text_input = st.text_area(label="Your comment", value="", height=200)
    if st.button("Analyze", key=2):
        if text_input.strip() == "":
            st.warning("Bạn chưa nhập đoạn văn bản")
        else:
            with st.spinner("Predicting..."):
                result = predict_class(text_input, model_setiment_analysis)
                st.write("Results:")
                if result == "Positive":
                    st.code("Positive 👍")
                else:
                    st.code("Negative 👎")
            st.success("Prediction completed!")




# Contact form
create_contact_form()
# Information app
create_infomation_app(name_app, version_app, current_time)