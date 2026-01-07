# app.py
# الملف الرئيسي للواجهة

import streamlit as st
import numpy as np
import os
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# استدعاء الملفات الأخرى
from utils import load_tflite_engine, tflite_predict, check_brightness
from database import CLASS_MAP, COSMIC_DATA, REDIRECT_MAP

# 1. إعداد الصفحة
st.set_page_config(page_title="المرصد الكوني", page_icon="🌌", layout="wide")
load_dotenv()

# --- التصميم CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    .main-header { text-align: center; color: #58a6ff; border-bottom: 2px solid #1f6feb; padding-bottom: 20px; }
    .result-box { background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d; text-align: center; }
    .stChatMessage { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔭 المرصد الكوني الذكي</h1>", unsafe_allow_html=True)

# القائمة الجانبية
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1909/1909848.png", width=100)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and "OPENAI_API_KEY" in st.secrets: api_key = st.secrets["OPENAI_API_KEY"]
    
    if api_key: st.success("✅ المساعد الذكي متصل")
    else: st.warning("⚠️ يعمل بوضع قاعدة البيانات فقط")

interpreter = load_tflite_engine()

# تهيئة الذاكرة
if "messages" not in st.session_state: st.session_state.messages = []
if "current_obj" not in st.session_state: st.session_state.current_obj = None

# المنطق الرئيسي
if interpreter:
    uploaded_file = st.file_uploader("📸 ارفع صورة الجرم السماوي", type=["jpg", "png", "jpeg", "jfif"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        # أ) التوقع
        preds = tflite_predict(interpreter, img)
        idx = np.argmax(preds)
        confidence = preds[idx] * 100
        
        # ب) الحصول على الاسم الخام
        raw_key = CLASS_MAP.get(idx, 'unknown')
        
        # ج) معالجة الشمس/العقرب (السطوع)
        brightness = check_brightness(img)
        if raw_key in ['scorpius', 'rocket', 'unknown', 'space_debris']:
            if brightness > 60:
                raw_key = 'sun'
                confidence = 99.9
        
        # د) التوجيه للبيانات الصحيحة
        data_key = raw_key
        if raw_key in REDIRECT_MAP:
            data_key = REDIRECT_MAP[raw_key]
            
        # هـ) جلب البيانات من database.py
        display_data = COSMIC_DATA.get(data_key, {
            'الاسم': raw_key.title(),
            'الحالة': 'بيانات تفصيلية غير متوفرة',
            'ملاحظة': 'اسأل المساعد الذكي للمزيد.'
        })
        
        arabic_title = display_data.get('الاسم', raw_key)
        
        # رسالة الشات التلقائية
        if st.session_state.current_obj != arabic_title:
            st.session_state.current_obj = arabic_title
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": f"تم رصد **{arabic_title}**. البيانات في الجدول."})

        # --- العرض ---
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(img, use_container_width=True)
            st.markdown(f"""
            <div class='result-box'>
                <h2 style='color:#58a6ff'>{arabic_title}</h2>
                <p>{raw_key}</p>
                <div style='color:#0f0'>الثقة: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("### 📊 البيانات العلمية (موثقة)")
            for k, v in display_data.items():
                if k != 'الاسم':
                    st.markdown(f"""
                    <div style='border-bottom:1px solid #333; padding:8px; display:flex; justify-content:space-between;'>
                        <span style='color:#8b949e'>{k}</span>
                        <span style='direction:ltr; color:#eee'>{v}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        
        # الشات
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.write(m["content"])
            
        if q := st.chat_input("سؤال إضافي؟"):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.write(q)
            with st.chat_message("assistant"):
                if api_key:
                    client = OpenAI(api_key=api_key)
                    p = f"الجرم: {arabic_title}. البيانات: {display_data}. السؤال: {q}.  أجب بالعربية."
                    try:
                        ans = client.chat.completions.create(model="gpt-4.1", messages=[{"role":"user", "content":p}]).choices[0].message.content
                        st.write(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                    except: st.error("خطأ اتصال")
                else:
                    st.warning("المساعد الذكي غير متصل.")