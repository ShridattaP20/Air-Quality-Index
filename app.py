import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


scaler=pickle.load(open('scaler.pkl','rb'))
rf_model=pickle.load(open('Air_quality_index.pkl','rb'))

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_aqi(PM25,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene,Xylene):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: PM2.5
        in: query
        type: number
        required: true
      - name: PM10
        in: query
        type: number
        required: true
      - name: NO
        in: query
        type: number
        required: true
      - name: NO2
        in: query
        type: number
        required: true
        - name: NOx
        in: query
        type: number
        required: true
        - name: NH3
        in: query
        type: number
        required: true
        - name: CO
        in: query
        type: number
        required: true
        - name: SO2
        in: query
        type: number
        required: true
        - name: O3
        in: query
        type: number
        required: true
        - name: Benzene
        in: query
        type: number
        required: true
        - name: Toluene
        in: query
        type: number
        required: true
        - name: Xylene
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    data=(PM25,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene,Xylene)
    data=scaler.transform(np.array(data).reshape(1,-1))
    prediction=rf_model.predict(data)
    print(prediction)
    return prediction



def main():
    st.title("AQI")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:left;">Streamlit AQI ML App </h2>
    </div>
     <div id="aqi-description" class="row col-md-10-x">
        <div class="col-md-10">
          <table class="table panel panel-default table-hover ">
            <thead>
              <tr class="active">
                <th class="col-md-1 text-center">AQI</th>
                <th class="col-md-2 text-center">Remark</th>
              </tr>
            </thead>
            <tbody class="aqi-desc-container">
            <tr class="" style="">
      <td class="col-md-1 aqi-value text-center text-middle">0-50</td>
      <td class="col-md-2 remark text-center text-middle">Good</td>
    </tr><tr class="" style="">
      <td class="col-md-1 aqi-value text-center text-middle">51-100</td>
      <td class="col-md-2 remark text-center text-middle">Satisfactory</td>
    </tr><tr class="" style="">
      <td class="col-md-1 aqi-value text-center text-middle">101-200</td>
      <td class="col-md-2 remark text-center text-middle">Moderate</td>
    </tr><tr class="" style="">
      <td class="col-md-1 aqi-value text-center text-middle">201-300</td>
      <td class="col-md-2 remark text-center text-middle">Poor</td>
    </tr><tr class="" style="">
      <td class="col-md-1 aqi-value text-center text-middle">301-400</td>
      <td class="col-md-2 remark text-center text-middle">Very Poor</td>
    </tr><tr class="" style="">
      <td class="col-md-1 aqi-value text-center text-middle">401-500</td>
      <td class="col-md-2 remark text-center text-middle">Severe</td>
    </tr></tbody>
          </table>
        </div>
        <hr>
      </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    PM25 = st.text_input("PM2.5","Type Here")
    PM10 = st.text_input("PM10","Type Here")
    NO = st.text_input("NO ","Type Here")
    NO2 = st.text_input("NO2","Type Here")
    NOx = st.text_input("NOx","Type Here")
    NH3 = st.text_input("NH3","Type Here")
    CO = st.text_input("CO","Type Here")
    SO2 = st.text_input("SO2","Type Here")
    O3 = st.text_input("O3","Type Here")
    Benzene = st.text_input("Benzene","Type Here")
    Toluene = st.text_input("Toluene ","Type Here")
    Xylene = st.text_input("Xylene","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_aqi(PM25,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene,Xylene)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")
    

if __name__=='__main__':
    main()
    