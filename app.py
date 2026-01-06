import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageOps
from dotenv import load_dotenv
import tensorflow as tf 
from openai import OpenAI

# 1. إعداد الصفحة
st.set_page_config(page_title="المرصد الكوني", page_icon="🌌", layout="wide")
load_dotenv()

# ========================================================
# 🗺️ الخريطة الرسمية (كما استخرجناها من كاجل)
# ========================================================
CLASS_MAP = {
    0: '51pegasib',
    1: 'aquarius',
    2: 'aries',
    3: 'white_dwarfs',
    4: 'asteroid_belt',
    5: 'spiral_galaxies',
    6: 'callisto',
    7: 'cancer',
    8: 'capricorn',
    9: 'carina',
    10: 'ceres',
    11: 'comets',
    12: 'crux',
    13: 'earth',
    14: 'elliptical_galaxies',
    15: 'giant_supergiant_stars',
    16: 'eris',
    19: 'europa',
    20: 'virgo',
    22: 'haumea',
    23: 'hydra',
    24: 'international_space_station',
    25: 'io',
    26: 'irregular_galaxies',
    27: 'jupiter',
    28: 'kepler10b',
    29: 'kepler22b',
    30: 'supermassive_black_holes',
    31: 'landers',
    32: 'leo',
    33: 'libra',
    35: 'mars',
    36: 'mercury',
    37: 'moon',
    38: 'neptune',
    40: 'orion',
    41: 'pisces',
    42: 'planetary_nebulae',
    43: 'pluto',
    44: 'probes',
    45: 'reflection_nebulae',
    46: 'rockets',
    47: 'rovers',
    48: 'sagittarius',
    49: 'satellites',
    50: 'saturn',
    51: 'scorpius',
    52: 'space_debris',
    54: 'neutron_stars_pulsars',
    55: 'sun',
    56: 'taurus',
    57: 'tiangong_space_station',
    58: 'titan',
    59: 'uranus',
    60: 'ursa_major_big_dipper_',
    61: 'venus',
    63: 'wasp12b',
}


# قاموس التعريب الشامل (أسماء + مصطلحات + أنواع + وحدات)
ARABIC_NAMES = {
    # --- الكواكب والأجرام ---
    'sun': 'الشمس', 'moon': 'القمر', 'mercury': 'عطارد', 'venus': 'الزهرة', 
    'earth': 'الأرض', 'mars': 'المريخ', 'jupiter': 'المشتري', 'saturn': 'زحل', 
    'uranus': 'أورانوس', 'neptune': 'نبتون', 'pluto': 'بلوتو',
    'black_hole': 'ثقب أسود', 'galaxy': 'مجرة', 'nebula': 'سديم',
    'asteroid': 'كويكب', 'asteroids': 'كويكبات', 'asteroid_belt': 'حزام الكويكبات',
    'comet': 'مذنب', 'comets': 'مذنبات', 'star': 'نجم',
    'international_space_station': 'محطة الفضاء الدولية',
    'rocket': 'صاروخ', 'satellite': 'قمر صناعي',
    'milky_way': 'درب التبانة', 'andromeda': 'أندروميدا',
    
    # --- المصطلحات (العناوين) ---
    'mass': 'الكتلة',
    'volume': 'الحجم',
    'radius': 'نصف القطر',
    'diameter': 'القطر',
    'density': 'الكثافة',
    'gravity': 'الجاذبية',
    'escape_velocity': 'سرعة الإفلات',
    'rotation_period': 'فترة الدوران',
    'orbital_period': 'فترة المدار',
    'mean_temp': 'متوسط الحرارة',
    'temperature': 'درجة الحرارة',
    'surface_pressure': 'ضغط السطح',
    'number_of_moons': 'عدد الأقمار',
    'moons': 'الأقمار',
    'distance': 'المسافة',
    'type': 'النوع',
    'discovery_date': 'تاريخ الاكتشاف',
    'discovered_by': 'المكتشف',
    'age': 'العمر',
    'atmosphere': 'الغلاف الجوي',

    # --- القيم والأنواع (Values) ---
    'terrestrial': 'كوكب صخري',
    'gas giant': 'عملاق غازي',
    'ice giant': 'عملاق جليدي',
    'dwarf planet': 'كوكب قزم',
    'spiral': 'حلزونية',
    'elliptical': 'إهليلجية',
    'irregular': 'غير منتظمة',
    'unknown': 'غير معروف',
    'confirmed': 'مؤكد',
    
    # --- الوحدات (Units) ---
    'kg': 'كغ',
    'km': 'كم',
    'days': 'يوم',
    'hours': 'ساعة',
    'years': 'سنة',
    'year': 'سنة',
    'au': 'وحدة فلكية',
    'light years': 'سنة ضوئية',
    'c': 'درجة مئوية',
    'k': 'كلفن'
}
# 2. تحميل البيانات والموديل
@st.cache_resource
def load_tflite_engine():
    try:
        # البحث عن أي ملف tflite في المجلد
        files = [f for f in os.listdir('.') if f.endswith('.tflite')]
        if files:
            interpreter = tf.lite.Interpreter(model_path=files[0])
            interpreter.allocate_tensors()
            return interpreter
        return None
    except: return None

def load_data():
    db = {}
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join("data", f))
                    df.columns = [c.strip() for c in df.columns]
                    db[f.replace('_info.csv', '')] = df
                except: pass
    return db

# 3. دالة التوقع (بدون قسمة - Raw Input 0-255)
# لأن النموذج يحتوي على طبقة Rescaling داخلية
def tflite_predict(interpreter, image):
    target_size = (224, 224)
    img_padded = ImageOps.pad(image, target_size, color='black')
    input_data = np.array(img_padded, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- الواجهة ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    .main-header { text-align: center; color: #58a6ff; border-bottom: 2px solid #1f6feb; padding-bottom: 20px; }
    .result-box { background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d; text-align: center; }
    .debug-info { font-family: monospace; color: #ffeb3b; font-size: 0.8em; direction: ltr; margin-top: 5px;}
    .stChatMessage { direction: rtl; text-align: right; }
    .stMarkdown p { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔭 المرصد الكوني الذكي</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1909/1909848.png", width=100)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key: st.success("✅ المفتاح متصل")
    
    with st.expander("ℹ️ عن النظام"):
        st.caption("يعتمد النظام على خريطة بيانات دقيقة مستخرجة من بيئة التدريب (Kaggle) لضمان دقة التصنيف.")

interpreter = load_tflite_engine()
csv_data = load_data()

# تهيئة الجلسة
if "messages" not in st.session_state: st.session_state.messages = []
if "current_obj" not in st.session_state: st.session_state.current_obj = None
if "obj_info" not in st.session_state: st.session_state.obj_info = {}

if interpreter:
    uploaded_file = st.file_uploader("📸 ارفع صورة الجرم السماوي", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        preds = tflite_predict(interpreter, img)
        idx = np.argmax(preds)
        confidence = preds[idx] * 100
        
        # 🔥 جلب الاسم الدقيق من الخريطة الرسمية 🔥
        if idx in CLASS_MAP:
            eng_name = CLASS_MAP[idx]
        else:
            eng_name = f"Unknown (ID: {idx})"
            
        # تعريب الاسم
        arabic_name = eng_name.replace("_", " ").title()
        for key, val in ARABIC_NAMES.items():
            if key in eng_name.lower():
                arabic_name = val; break
        
        # تحديث الجلسة والبحث عن المعلومات
        if st.session_state.current_obj != eng_name:
            st.session_state.current_obj = eng_name
            st.session_state.messages = []
            info = {}
            
            # البحث الذكي في ملفات CSV
            # نبحث عن الاسم الانجليزي (أول كلمة منه لتجنب الفروقات مثل _belt)
            search_key = eng_name.lower().split('_')[0]
            
            # حالات خاصة للبحث
            if eng_name == '51pegasib': search_key = 'pegasi'
            
            for k, df in csv_data.items():
                # نبحث في أول عمود (Name)
                match = df[df.iloc[:,0].astype(str).str.lower().str.contains(search_key, regex=False)]
                if not match.empty:
                    # نأخذ أول نتيجة ونحولها لقاموس
                    info = {x: str(y) for i, (x,y) in enumerate(match.iloc[0].items()) if i<8}
                    break
            
            st.session_state.obj_info = info
            
            # رسالة الترحيب
            welcome_msg = f"تم رصد **{arabic_name}** ({eng_name})."
            if confidence < 50:
                welcome_msg += " (نسبة التأكد منخفضة، قد تكون الصورة غير واضحة)."
            else:
                welcome_msg += " أنا جاهز للإجابة على أسئلتك."
                
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

        # العرض
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(img, caption="الصورة المحللة", use_container_width=True)
            st.markdown(f"""
            <div class='result-box'>
                <h2 style='color:#58a6ff'>{arabic_name}</h2>
                <div class='debug-info'>ID: {idx} | Name: {eng_name}</div>
                <div style='color:#3fb950; font-weight:bold; margin-top:5px'>الثقة: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        # ... داخل العمود الثاني (with c2:) ...
        with c2:
            st.markdown("### 📊 البيانات العلمية")
            if st.session_state.obj_info:
                for k, v in st.session_state.obj_info.items():
                    # 1. تعريب العنوان (Key)
                    key_lower = str(k).lower()
                    arabic_key = k # الافتراضي
                    for en, ar in ARABIC_NAMES.items():
                        if en in key_lower:
                            arabic_key = ar
                            break
                    
                    # 2. تعريب القيمة (Value) - السحر هنا ✨
                    val_str = str(v)
                    arabic_value = val_str # الافتراضي
                    
                    # نفحص كل كلمة في القيمة ونحاول تعريبها
                    # هذا سيحول "Terrestrial Planet" إلى "كوكب صخري"
                    # ويحول "365 days" إلى "365 يوم"
                    for en, ar in ARABIC_NAMES.items():
                        # نستخدم مسافات لضمان عدم استبدال حروف وسط الكلمة
                        # (مثلاً لا نستبدل kg داخل wordkg)
                        if f" {en} " in f" {val_str.lower()} ": 
                            # استبدال مع الحفاظ على حالة الأحرف (Case Insensitive Replace)
                            import re
                            arabic_value = re.sub(re.escape(en), ar, arabic_value, flags=re.IGNORECASE)
                    
                    # تنسيق العرض (جعل الأرقام يسار لليمين للقراءة الصحيحة، والنص عربي)
                    st.markdown(f"""
                    <div style='border-bottom:1px solid #333; padding:8px; display:flex; justify-content:space-between; align-items:center;'>
                        <span style='color:#8b949e; font-weight:bold; font-size:0.95em;'>{arabic_key}</span>
                        <span style='color:#eee; direction:ltr; unicode-bidi: embed;'>{arabic_value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("البيانات الأرشيفية غير متوفرة، اسأل المساعد الذكي.")
        st.markdown("---")
        
        # الشات
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.write(m["content"])
            
        if q := st.chat_input("اكتب سؤالك هنا..."):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.write(q)
            with st.chat_message("assistant"):
                if api_key:
                    client = OpenAI(api_key=api_key)
                    with st.spinner("جاري التحليل..."):
                        p = f"""
                        أنت خبير فلكي.
                        الجرم: {arabic_name} ({eng_name}).
                        البيانات: {st.session_state.obj_info}.
                        السؤال: {q}
                        أجب باللغة العربية.
                        """
                        try:
                            ans = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":p}]).choices[0].message.content
                            st.write(ans)
                            st.session_state.messages.append({"role": "assistant", "content": ans})
                        except: st.error("خطأ اتصال")
                else: st.error("المفتاح مفقود")