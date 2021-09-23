import pandas as pd
import streamlit as st

from MLUtils import *

st.title('AI, OS and RE for LUAD')  # 算法名称 and XXX

gnb_list = []
adab_list = []

FIXED_TRAIN_FILENAME = "train_OS_RE.csv"
FIXED_TRAIN_Y_NUM = 2
COL_INPUT = [
    "hospitalization days", "Stage", "Smoking", "Aging", "weight", "Duration of anesthesia", "thoracoscope",
    "Bleeding volume", "Urine", "platelet count", "Neutrophil", "Lymphocyte", "NLR", "PLR", "Monocyte", "LMR",
    "Eosinophil", "Basophil", "thrombocytocrit", "platelet distribution width"
]
COL_Y = []
vars = []
btn_predict = None


# 配置选择变量（添加生成新数据并预测的功能）
def setup_selectors():
    global vars, btn_predict

    if COL_INPUT is not None and len(COL_INPUT) > 0:
        col_num = 3
        cols = st.columns(col_num)

        for i, c in enumerate(COL_INPUT):
            with cols[i % col_num]:
                num_input = st.number_input(f"Please input {c}", value=0, format="%d", key=c)
                vars.append(num_input)

        with cols[0]:
            btn_predict = st.button("Do Predict")

    if btn_predict:
        do_predict()


# 对上传的文件进行处理和展示
def do_processing():
    global COL_Y
    global gnb_list, adab_list
    pocd_list = read_csv(FIXED_TRAIN_FILENAME, y_num=FIXED_TRAIN_Y_NUM)
    # print(pocd_list)  # debug
    # print(COL_Y)  # debug

    for np, pocd in enumerate(pocd_list):
        Y = str(pocd.columns[-1])
        COL_Y.append(Y)
        st.text(f"Dataset Description (Y: {Y})")
        st.write(pocd.describe())
        if st.checkbox('Show detail of this dataset', key=f"cb_show_detail_{Y.lower()}"):
            st.write(pocd)

        # 分割数据
        X_train, X_test, y_train, y_test = do_split_data(pocd)
        X_train, X_test, y_train, y_test = do_xy_preprocessing(X_train, X_test, y_train, y_test)

        col1, col2 = st.columns(2)

        # 准备模型
        gnb = GaussianNB()
        gnb_list.append(gnb)
        adab = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=1)
        adab_list.append(adab)

        # 模型训练、显示结果
        with st.spinner("Training, please wait..."):
            gnb_result = model_fit_score(gnb, X_train, y_train)
            adab_result = model_fit_score(adab, X_train, y_train)
        with col1:
            st.text("Training Result")
            msg = model_print(gnb_result, "GaussianNB - Train")
            st.write(msg)
            msg = model_print(adab_result, "AdaBoost - Train")
            st.write(msg)
            # 训练画图
            fig_train = plt_roc_auc([
                (gnb_result, 'GaussianNB',),
                (adab_result, 'AdaBoost',),
            ], 'Train ROC')
            st.pyplot(fig_train)
        # 模型测试、显示结果
        with st.spinner("Testing, please wait..."):
            gnb_test_result = model_score(gnb, X_test, y_test)
            adab_test_result = model_score(adab, X_test, y_test)
        with col2:
            st.text("Testing Result")
            msg = model_print(gnb_test_result, "GaussianNB - Test")
            st.write(msg)
            msg = model_print(adab_test_result, "AdaBoost - Test")
            st.write(msg)
            # 测试画图
            fig_test = plt_roc_auc([
                (gnb_test_result, 'GaussianNB',),
                (adab_test_result, 'AdaBoost',),
            ], 'Validate ROC')
            st.pyplot(fig_test)


# 对生成的预测数据进行处理
def do_predict():
    global COL_Y, vars
    global gnb_list, adab_list

    # 处理生成的预测数据的输入
    pocd_predict = pd.DataFrame(data=[vars], columns=COL_INPUT)
    pocd_predict = do_base_preprocessing(pocd_predict, with_y=False, y_num=FIXED_TRAIN_Y_NUM)
    st.text("Preview for detail of this predict data")
    st.write(pocd_predict)
    pocd_predict = do_predict_preprocessing(pocd_predict)

    # 进行预测并输出
    for ng, gnb in enumerate(gnb_list):
        Y = COL_Y[ng]
        # GaussianNB
        pr = gnb.predict(pocd_predict)
        pr = pr.astype(np.int)
        st.markdown(r"$\color{red}{GaussianNB}$ $\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{" + str(
            Y) + r"}$ $\color{red}{is}$ $\color{red}{" + str(pr[0]) + "}$")
    for ng, adab in enumerate(adab_list):
        Y = COL_Y[ng]
        # AdaBoost
        pr = adab.predict(pocd_predict)
        pr = pr.astype(np.int)
        st.markdown(r"$\color{red}{AdaBoost}$ $\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{" + str(
            Y) + r"}$ $\color{red}{is}$ $\color{red}{" + str(pr[0]) + "}$")


if __name__ == "__main__":
    do_processing()
    setup_selectors()
