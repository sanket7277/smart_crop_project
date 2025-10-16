import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import requests
import time

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Configuration ---
OPENWEATHER_API_KEY = "649b57f4af88bbe329b17952ec27b162"

if not OPENWEATHER_API_KEY or len(OPENWEATHER_API_KEY) != 32:
    OPENWEATHER_API_KEY = None
    st.error("API कुंजी गुम है या गलत लंबाई की है। लाइव मौसम सुविधाएँ अक्षम हैं।")


# ==============================================================================
# --- 0. Multilanguage Translations (i18n) ---
# ==============================================================================

# Dictionary for translating the final CROP NAMES from English (ML output)
CROP_NAME_TRANSLATIONS = {
    'Banana': {'Hindi': 'केला', 'Marathi': 'केळी'},
    'Chickpea': {'Hindi': 'चना', 'Marathi': 'चणा'},
    'Mungbean': {'Hindi': 'मूंग', 'Marathi': 'मूग'},
    'Apple': {'Hindi': 'सेब', 'Marathi': 'सफरचंद'},
    'Cotton': {'Hindi': 'कपास', 'Marathi': 'कापूस'},
    'Mothbeans': {'Hindi': 'मोठ', 'Marathi': 'मठ'},
    'Grapes': {'Hindi': 'अंगूर', 'Marathi': 'द्राक्षे'},
    'Mango': {'Hindi': 'आम', 'Marathi': 'आंबा'},
    'Coconut': {'Hindi': 'नारियल', 'Marathi': 'नारळ'},
    'Jute': {'Hindi': 'जूट', 'Marathi': 'ताग'},
    'Lentil': {'Hindi': 'मसूर', 'Marathi': 'मसूर'},
    'Blackgram': {'Hindi': 'उड़द', 'Marathi': 'उडीद'},
    'Coffee': {'Hindi': 'कॉफी', 'Marathi': 'कॉफी'},
    'Kidneybeans': {'Hindi': 'राजमा', 'Marathi': 'राजमा'},
    'Maize': {'Hindi': 'मक्का', 'Marathi': 'मका'}
}

# Dictionary for translating the descriptive TIME STRINGS (NEW FIX)
TIME_DETAILS_TRANSLATIONS = {
    "Feb - May or July - Aug (Planting)": {"Hindi": "फरवरी - मई या जुलाई - अगस्त (रोपण)", "Marathi": "फेब्रुवारी - मे किंवा जुलै - ऑगस्ट (लागवड)"},
    "October - November (Rabi)": {"Hindi": "अक्टूबर - नवंबर (रबी)", "Marathi": "ऑक्टोबर - नोव्हेंबर (रब्बी)"},
    "June - July (Kharif)": {"Hindi": "जून - जुलाई (खरीफ)", "Marathi": "जून - जुलै (खरीप)"},
    "Dec - Feb (Dormant season)": {"Hindi": "दिसंबर - फरवरी (निष्क्रिय मौसम)", "Marathi": "डिसेंबर - फेब्रुवारी (निष्क्रिय हंगाम)"},
    "June - September (Kharif)": {"Hindi": "जून - सितंबर (खरीफ)", "Marathi": "जून - सप्टेंबर (खरीप)"},
    "Dec - Jan (Planting)": {"Hindi": "दिसंबर - जनवरी (रोपण)", "Marathi": "डिसेंबर - जानेवारी (लागवड)"},
    "July - August (Planting)": {"Hindi": "जुलाई - अगस्त (रोपण)", "Marathi": "जुलै - ऑगस्ट (लागवड)"},
    "June - July (Planting)": {"Hindi": "जून - जुलै (रोपण)", "Marathi": "जून - जुलै (लागवड)"},
    "March - July (Monsoon)": {"Hindi": "मार्च - जुलाई (मानसून)", "Marathi": "मार्च - जुलै (मान्सून)"},
    "June - July (Monsoon/Main planting)": {"Hindi": "जून - जुलाई (मानसून/मुख्य रोपण)", "Marathi": "जून - जुलै (मान्सून/मुख्य लागवड)"},
    "Oct - Nov (Rabi) or Feb-Mar (Zaid)": {"Hindi": "अक्टूबर - नवंबर (रबी) या फरवरी - मार्च (जायद)", "Marathi": "ऑक्टोबर - नोव्हेंबर (रब्बी) किंवा फेब्रुवारी - मार्च (जायद)"},
    "11 - 15 months": {"Hindi": "11 - 15 महीने", "Marathi": "11 - 15 महिने"},
    "4 - 5 months": {"Hindi": "4 - 5 महीने", "Marathi": "4 - 5 महिने"},
    "2 - 3 months": {"Hindi": "2 - 3 महीने", "Marathi": "2 - 3 महिने"},
    "Perennial (Tree)": {"Hindi": "बारहमासी (वृक्ष)", "Marathi": "बारमाही (झाड)"},
    "5 - 6 months": {"Hindi": "5 - 6 महीने", "Marathi": "5 - 6 महिने"},
    "Perennial (Vine)": {"Hindi": "बारहमासी (बेल)", "Marathi": "बारमाही (वेल)"},
    "Perennial (Shrub)": {"Hindi": "बारहमासी (झाड़ी)", "Marathi": "बारमाही (झुडूप)"},
    "3 - 5 months": {"Hindi": "3 - 5 महीने", "Marathi": "3 - 5 महिने"},
    "3 - 4 months": {"Hindi": "3 - 4 महीने", "Marathi": "3 - 4 महिने"},
    "Throughout the year": {"Hindi": "साल भर", "Marathi": "वर्षभर"},
    "March (Spring)": {"Hindi": "मार्च (वसंत)", "Marathi": "मार्च (वसंत)"},
    "September - October (Autumn)": {"Hindi": "सितंबर - अक्टूबर (शरद ऋतु)", "Marathi": "सप्टेंबर - ऑक्टोबर (शरद ऋतू)"},
    "July - September": {"Hindi": "जुलाई - सितंबर", "Marathi": "जुलै - सप्टेंबर"},
    "December - February": {"Hindi": "दिसंबर - फरवरी", "Marathi": "डिसेंबर - फेब्रुवारी"},
    "Feb - May": {"Hindi": "फरवरी - मई", "Marathi": "फेब्रुवारी - मे"},
    "February - June": {"Hindi": "फरवरी - जून", "Marathi": "फेब्रुवारी - जून"},
    "Year-round (Multiple harvests)": {"Hindi": "साल भर (कई फसलें)", "Marathi": "वर्षभर (एकाधिक कापणी)"},
    "July - October (Autumn)": {"Hindi": "जुलाई - अक्टूबर (शरद ऋतु)", "Marathi": "जुलै - ऑक्टोबर (शरद ऋतू)"},
    "February - April (Spring)": {"Hindi": "फरवरी - अप्रैल (वसंत)", "Marathi": "फेब्रुवारी - एप्रिल (वसंत)"},
    "November - March (Main Harvest)": {"Hindi": "नवंबर - मार्च (मुख्य कटाई)", "Marathi": "नोव्हेंबर - मार्च (मुख्य कापणी)"},
    "March - May (Spring/Early Summer)": {"Hindi": "मार्च - मई (वसंत/जल्दी गर्मी)", "Marathi": "मार्च - मे (वसंत/लवकर उन्हाळा)"}
}


LANGUAGES = {
    'English': {
        'code': 'English', 
        'title': "🌱 Smart Crop Recommendation System",
        'intro': "Enter your soil and climate parameters below. The recommendation engine uses these values to predict the best crop.",
        'sidebar_header': "Soil & Climate Parameters",
        'weather_subheader': "Live Weather Fetch (Auto)",
        'city_label': "City Name (e.g., Pune)",
        'n_label': "Nitrogen (N) kg/ha",
        'p_label': "Phosphorous (P) kg/ha",
        'k_label': "Potassium (K) kg/ha",
        'temp_label': "Current Temperature (°C)",
        'hum_label': "Current Humidity (%)",
        'ph_label': "Soil pH",
        'rain_label': "Rainfall (mm) - Long-term Avg",
        'soil_subheader': "Soil Type (Must Match ML Model)",
        'soil_label': "Select Soil Type",
        'soil_types': ["Alluvial", "Black (Regur)", "Red & Yellow", "Laterite", "Arid (Desert)", "Forest / Mountain", "Saline / Alkaline", "Peaty / Marshy"],
        'loc_subheader': "Location Data (Optional)",
        'lat_label': "Latitude",
        'lon_label': "Longitude",
        'button_text': "Recommend Crop",
        'rec_header': "Optimal Crop Recommended:",
        'rec_base_text': "Based on the provided soil and climate conditions, <b>{}</b> is the most suitable crop.",
        'time_subheader': "🗓️ Best Time Details",
        'sowing_time': "Best Sowing Time",
        'duration_time': "Duration Time (Approx.)",
        'harvest_time': "Harvesting Time",
        'note': "Note: This recommendation is based on a machine learning model trained on the provided dataset and real-time/default climate data. **The ML Model MUST be retrained to include the Soil Type feature.**",
        'error_model': "Error: 'model.pkl' not found. Please run the model training notebook first.",
        'error_map': "Error: 'Crop_data.csv' not found. Please ensure it exists in the project folder to get crop details.", 
        'error_pred': "Prediction failed: {}",
        'info_pred': "Ensure your 'model.pkl' and 'Crop_data.csv' files are correctly set up.",
        'unknown_crop': "Unknown Crop",
        'not_available': "Not available in current data.",
        'live_weather_success': "✅ Live data fetched! Temperature: **{temp:.1f}°C** and Humidity: **{hum}%**.",
        'live_weather_error': "❌ Could not fetch live data. Using manual **{temp:.1f}°C** / **{hum:.0f}%**. Error: {error}",
        'weather_disabled': "Live weather features disabled.",
    },
    'हिन्दी (Hindi)': {
        'code': 'Hindi',
        'title': "🌱 स्मार्ट फसल सुझाव प्रणाली",
        'intro': "अपनी मिट्टी और जलवायु पैरामीटर नीचे दर्ज करें। सुझाव इंजन इन मूल्यों का उपयोग करके सर्वोत्तम फसल की भविष्यवाणी करता है।",
        'sidebar_header': "मिट्टी और जलवायु पैरामीटर",
        'weather_subheader': "लाइव मौसम प्राप्त करें (स्वचालित)",
        'city_label': "शहर का नाम (जैसे, पुणे)",
        'n_label': "नाइट्रोजन (N) किग्रा/हेक्टेयर",
        'p_label': "फास्फोरस (P) किग्रा/हेक्टेयर",
        'k_label': "पोटेशियम (K) किग्रा/हेक्टेयर",
        'temp_label': "वर्तमान तापमान (°C)",
        'hum_label': "वर्तमान नमी (%)",
        'ph_label': "मिट्टी का पीएच (pH)",
        'rain_label': "वर्षा (मिमी) - दीर्घकालिक औसत",
        'soil_subheader': "मिट्टी का प्रकार (ML मॉडल से मेल खाना चाहिए)",
        'soil_label': "मिट्टी का प्रकार चुनें",
        'soil_types': ["जलोढ़", "काली (रेगुर)", "लाल और पीली", "लेटराइट", "शुष्क (रेगिस्तानी)", "वन / पर्वतीय", "खारी / क्षारीय", "दलदली / पीट"],
        'loc_subheader': "स्थान डेटा (वैकल्पिक)",
        'lat_label': "अक्षांश",
        'lon_label': "देशांतर",
        'button_text': "फसल सुझाएँ",
        'rec_header': "इष्टतम फसल सुझाव:",
        'rec_base_text': "प्रदान की गई मिट्टी और जलवायु परिस्थितियों के आधार पर, <b>{}</b> सबसे उपयुक्त फसल है।",
        'time_subheader': "🗓️ सर्वोत्तम समय विवरण",
        'sowing_time': "बुवाई का सर्वोत्तम समय",
        'duration_time': "अवधि का समय (लगभग)",
        'harvest_time': "कटाई का समय",
        'note': "ध्यान दें: यह सुझाव प्रदान किए गए डेटासेट और वास्तविक समय/डिफ़ॉल्ट जलवायु डेटा पर प्रशिक्षित मशीन लर्निंग मॉडल पर आधारित है। **ML मॉडल को मिट्टी के प्रकार की सुविधा को शामिल करने के लिए पुन: प्रशिक्षित किया जाना चाहिए।**",
        'error_model': "त्रुटी: 'model.pkl' नहीं मिला। कृपया पहले मॉडेल प्रशिक्षण नोटबुक चलाएँ।",
        'error_map': "त्रुटी: 'Crop_data.csv' नहीं मिला। सुनिश्चित करें कि यह प्रोजेक्ट फ़ोल्डर में मौजूद है ताकि फसल का विवरण मिल सके।", 
        'error_pred': "भविष्यवाणी विफल: {}",
        'info_pred': "सुनिश्चित करें कि आपकी 'model.pkl' और 'Crop_data.csv' फ़ाइलें सही ढंग से सेट हैं।",
        'unknown_crop': "अज्ञात फसल",
        'not_available': "वर्तमान डेटा में उपलब्ध नहीं है।",
        'live_weather_success': "✅ लाइव डेटा प्राप्त हुआ! तापमान: **{temp:.1f}°C** और नमी: **{hum}%**।",
        'live_weather_error': "❌ लाइव डेटा प्राप्त नहीं हो सका। इसके बजाय मैनुअल **{temp:.1f}°C** / **{hum:.0f}%** का उपयोग कर रहा हूँ। त्रुटि: {error}",
        'weather_disabled': "लाइव मौसम सुविधाएँ अक्षम हैं।",
    },
    'मराठी (Marathi)': {
        'code': 'Marathi',
        'title': "🌱 स्मार्ट पीक शिफारस प्रणाली",
        'intro': "तुमच्या जमिनीसाठी योग्य पीक शिफारस मिळवण्यासाठी माती आणि हवामानाचे पॅरामीटर्स खाली प्रविष्ट करा। शिफारस इंजिन या मूल्यांचा वापर करून सर्वोत्तम पिकाचा अंदाज लावते।",
        'sidebar_header': "माती आणि हवामान पॅरामीटर्स",
        'weather_subheader': "थेट हवामान प्राप्त करा (स्वयंचलित)",
        'city_label': "शहराचे नाव (उदा. पुणे)",
        'n_label': "नत्र (N) किग्रॅ/हेक्टर",
        'p_label': "फॉस्फरस (P) किग्रॅ/हेक्टर",
        'k_label': "पोटॅशियम (K) किग्रॅ/हेक्टर",
        'temp_label': "सध्याचे तापमान (°C)",
        'hum_label': "सध्याची आर्द्रता (%)",
        'ph_label': "मातीचा पीएच (pH)",
        'rain_label': "पाऊस (मिमी) - दीर्घकालीन सरासरी",
        'soil_subheader': "मातीचा प्रकार (ML मॉडेलशी जुळला पाहिजे)",
        'soil_label': "मातीचा प्रकार निवडा",
        'soil_types': ["गाळाची", "काळी (रेगुर)", "लाल आणि पिवळी", "जांभा", "कोरडी (वाळवंटी)", "वन / पर्वतीय", "खारट / अल्कधर्मी", "दलदलीचा / पीटी"],
        'loc_subheader': "स्थान डेटा (ऐच्छिक)",
        'lat_label': "अक्षांश",
        'lon_label': "रेखांश",
        'button_text': "पीक सुचवा",
        'rec_header': "शिफारस केलेले इष्टतम पीक:",
        'rec_base_text': "प्रदान केलेल्या माती आणि हवामान परिस्थितीनुसार, <b>{}</b> हे सर्वात योग्य पीक आहे।",
        'time_subheader': "🗓️ सर्वोत्तम वेळेचा तपशील",
        'sowing_time': "पेरणीची सर्वोत्तम वेळ",
        'duration_time': "कालावधी (अंदाजे)",
        'harvest_time': "कापणीची वेळ",
        'note': "टीप: ही शिफारस प्रदान केलेल्या डेटासेट आणि वास्तविक वेळेतील/डिफॉल्ट हवामान डेटावर प्रशिक्षित मशीन लर्निंग मॉडेलवर आधारित आहे। **ML मॉडेलला मातीचा प्रकार वैशिष्ट्य समाविष्ट करण्यासाठी पुन्हा प्रशिक्षित करणे आवश्यक आहे।**",
        'error_model': "त्रुटी: 'model.pkl' सापडले नाही। कृपया प्रथम मॉडेल प्रशिक्षण नोटबुक चालवा।",
        'error_map': "त्रुटी: 'Crop_data.csv' सापडले नाही। कृपया ते प्रोजेक्ट फोल्डरमध्ये असल्याची खात्री करा जेणेकरून पिकाचा तपशील मिळेल।", 
        'error_pred': "अंदाज अयशस्वी: {}",
        'info_pred': "सुनिश्चित करा की तुमची 'model.pkl' आणि 'Crop_data.csv' फ़ाइल योग्यरित्या सेट केल्या आहेत याची खात्री करा।",
        'unknown_crop': "अज्ञात पीक",
        'not_available': "वर्तमान डेटा मध्ये उपलब्ध नाही।",
        'live_weather_success': "✅ थेट डेटा प्राप्त झाला! तापमान: **{temp:.1f}°C** आणि आर्द्रता: **{hum}%**।",
        'live_weather_error': "❌ थेट डेटा प्राप्त होऊ शकला नाही। त्याऐवजी मैनुअल **{temp:.1f}°C** / **{hum:.0f}%** चा वापर करत आहे। त्रुटि: {error}",
        'weather_disabled': "लाइव मौसम सुविधाएँ अक्षम आहेत।",
    }
}


# --- Language Selection ---
with st.sidebar:
    selected_language = st.selectbox("Select Language | भाषा निवडा", list(LANGUAGES.keys()), index=0)
    T = LANGUAGES[selected_language] # Translation dictionary for the selected language
    LANG_CODE = T.get('code', 'English') # Get the language code for result translation


# --- OpenWeatherMap API Functions (Current Weather 2.5 - Free Tier) ---
@st.cache_data(ttl=600) 
def get_live_weather(city_name, api_key):
    """Fetches current temperature (Celsius) and humidity (%) from OpenWeatherMap 2.5 API."""
    if not city_name or not api_key:
        return None, None, "City name or API key missing."
        
    try:
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
        response = requests.get(complete_url)
        data = response.json()

        if response.status_code == 200:
            live_temp = data['main']['temp']
            live_hum = data['main']['humidity']
            return live_temp, live_hum, None
        else:
            error_message = data.get("message", f"API Error (Code {response.status_code}).")
            return None, None, error_message

    except requests.exceptions.RequestException as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, str(e)

# --- Constants and Defaults ---
DEFAULT_LAT = 20.0
DEFAULT_LON = 75.0
DEFAULT_TEMP = 25.0
DEFAULT_HUM = 65.0
DEFAULT_RAINFALL = 150.0

def get_past_weather_simulation():
    return {'rainfall': DEFAULT_RAINFALL}

# --- CROP TIME DETAILS (English Keys for Lookup) ---
CROP_TIME_DETAILS = {
    'Banana': {'Sowing': 'Feb - May or July - Aug (Planting)', 'Duration': '11 - 15 months', 'Harvest': 'Throughout the year'},
    'Chickpea': {'Sowing': 'October - November (Rabi)', 'Duration': '4 - 5 months', 'Harvest': 'March (Spring)'},
    'Mungbean': {'Sowing': 'June - July (Kharif)', 'Duration': '2 - 3 months', 'Harvest': 'September - October (Autumn)'},
    'Apple': {'Sowing': 'Dec - Feb (Dormant season)', 'Duration': 'Perennial (Tree)', 'Harvest': 'July - September'},
    'Cotton': {'Sowing': 'June - September (Kharif)', 'Duration': '5 - 6 months', 'Harvest': 'December - February'},
    'Mothbeans': {'Sowing': 'June - July (Kharif)', 'Duration': '2 - 3 months', 'Harvest': 'September - October (Autumn)'},
    'Grapes': {'Sowing': 'Dec - Jan (Planting)', 'Duration': 'Perennial (Vine)', 'Harvest': 'Feb - May'},
    'Mango': {'Sowing': 'July - August (Planting)', 'Duration': 'Perennial (Tree)', 'Harvest': 'February - June'},
    'Coconut': {'Sowing': 'June - July (Planting)', 'Duration': 'Perennial (Tree)', 'Harvest': 'Year-round (Multiple harvests)'},
    'Jute': {'Sowing': 'March - July (Monsoon)', 'Duration': '4 - 5 months', 'Harvest': 'July - October (Autumn)'},
    'Lentil': {'Sowing': 'October - November (Rabi)', 'Duration': '4 - 5 months', 'Harvest': 'February - April (Spring)'},
    'Blackgram': {'Sowing': 'June - July (Kharif)', 'Duration': '2 - 3 months', 'Harvest': 'September - October (Autumn)'},
    'Coffee': {'Sowing': 'June - July (Monsoon/Main planting)', 'Duration': 'Perennial (Shrub)', 'Harvest': 'November - March (Main Harvest)'},
    'Kidneybeans': {'Sowing': 'Oct - Nov (Rabi) or Feb-Mar (Zaid)', 'Duration': '3 - 5 months', 'Harvest': 'March - May (Spring/Early Summer)'},
    'Maize': {'Sowing': 'June - July (Kharif)', 'Duration': '3 - 4 months', 'Harvest': 'September - October (Autumn)'}
}


# --- 1. Load Model and Crop Mapping ---
@st.cache_resource
def load_assets():
    """Load the trained ML model and create the crop mapping."""
    try:
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        st.error(T['error_model'])
        return None, None
    except Exception as e:
        st.error(f"{T['error_model']}: {e}")
        return None, None

    try:
        df_original = pd.read_csv('Crop_data.csv') 
        le = LabelEncoder()
        le.fit(df_original['Crop'])
        crop_map = dict(zip(le.transform(le.classes_), le.classes_))
        return model, crop_map
    except FileNotFoundError:
        st.error(T['error_map'])
        return model, None
    except Exception as e:
        st.error(f"{T['error_map']}: {e}")
        return model, None

model, crop_map = load_assets()

if model is None or crop_map is None:
    st.stop()


# --- 2. Sidebar Input Setup ---

st.sidebar.header(T['sidebar_header'])

with st.sidebar:
    # --- 2A. Live Weather Inputs ---
    st.subheader(T['weather_subheader'])
    city_name = st.text_input(T['city_label'], key="city_input", value="Pune")
    
    live_temp, live_hum, weather_error = None, None, None

    if OPENWEATHER_API_KEY:
        live_temp, live_hum, weather_error = get_live_weather(city_name, OPENWEATHER_API_KEY)

        if live_temp is not None:
            st.success(T['live_weather_success'].format(temp=live_temp, hum=live_hum))
        else:
            st.warning(T['live_weather_error'].format(error=weather_error, temp=DEFAULT_TEMP, hum=DEFAULT_HUM))
    else:
        st.info(T['weather_disabled'])


# --- 2B. Final Parameter Setup (Using Live Data or Default) ---

final_temp = live_temp if live_temp is not None else DEFAULT_TEMP
final_hum = live_hum if live_hum is not None else DEFAULT_HUM

st.markdown("---")

with st.sidebar:
    st.markdown(f"**Temperature used for ML:** **{final_temp:.1f}°C**")
    st.markdown(f"**Humidity used for ML:** **{final_hum:.0f}%**")
    st.markdown("---")
    
    # Manual Sliders
    st.subheader("Soil Nutrients")
    N = st.slider(T['n_label'], 0, 140, 70, key='N_in')
    P = st.slider(T['p_label'], 0, 150, 45, key='P_in')
    K = st.slider(T['k_label'], 0, 200, 90, key='K_in')
    ph = st.slider(T['ph_label'], 3.5, 9.5, 6.5, key='pH_in')
    
    # Rainfall
    past_data = get_past_weather_simulation() 
    rainfall = st.slider(T['rain_label'], 20.0, 300.0, past_data['rainfall'], key='R_in')
    
    # --- Soil Type Input ---
    st.subheader(T['soil_subheader'])
    soil_types_list = T['soil_types']
    soil_type_selected_display = st.selectbox(T['soil_label'], soil_types_list, key='Soil_Type_in')
    
    soil_type_index = soil_types_list.index(soil_type_selected_display)
    soil_type_english = LANGUAGES['English']['soil_types'][soil_type_index]
    
    # Lat/Lon are optional
    st.subheader(T['loc_subheader'])
    latitude = st.slider(T['lat_label'], 10.0, 30.0, DEFAULT_LAT, step=0.0001, format="%.4f", key='lat')
    longitude = st.slider(T['lon_label'], 70.0, 80.0, DEFAULT_LON, step=0.0001, format="%.4f", key='lon')


# --- 3. Main UI and Prediction ---
st.title(T['title'])
st.markdown(T['intro'])
st.markdown("---")

if st.button(T['button_text']):
    
    # --- Soil Type Encoding Logic ---
    soil_type_map = {
        "Alluvial": 1, 
        "Black (Regur)": 2, 
        "Red & Yellow": 3, 
        "Laterite": 4, 
        "Arid (Desert)": 5, 
        "Forest / Mountain": 6, 
        "Saline / Alkaline": 7, 
        "Peaty / Marshy": 8
    }
    
    soil_type_encoded = soil_type_map.get(soil_type_english, 0)
    
    # --- Input Data for ML (10 features) ---
    input_data = pd.DataFrame([[N, P, K, final_temp, final_hum, ph, rainfall, latitude, longitude, soil_type_encoded]],
                              columns=['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Latitude', 'Longitude', 'Soil_Type_Encoded'])
    
    try:
        prediction_encoded = model.predict(input_data)[0]
        
        # Predicted crop name in ENGLISH (from the ML model/crop_map)
        predicted_crop_english = crop_map.get(prediction_encoded, 'Unknown Crop')

        # --- DYNAMIC TRANSLATION LOGIC (FIX FOR CROP NAME) ---
        if LANG_CODE != 'English' and predicted_crop_english in CROP_NAME_TRANSLATIONS:
            predicted_crop_display = CROP_NAME_TRANSLATIONS[predicted_crop_english].get(LANG_CODE, predicted_crop_english)
        else:
            predicted_crop_display = predicted_crop_english
        
        
        # Retrieve the added time details (English keys for lookup)
        crop_details_english_keys = CROP_TIME_DETAILS.get(predicted_crop_english, {
            'Sowing': 'Not available in current data.',
            'Duration': 'Not available in current data.',
            'Harvest': 'Not available in current data.'
        })
        
        # --- DYNAMIC TRANSLATION LOGIC (FIX FOR TIME DETAILS) ---
        crop_details = {}
        for key, english_value in crop_details_english_keys.items():
            if LANG_CODE != 'English' and english_value in TIME_DETAILS_TRANSLATIONS:
                # Look up the translation for the time string
                translated_value = TIME_DETAILS_TRANSLATIONS[english_value].get(LANG_CODE, english_value)
            else:
                # Use the English value if no translation or language is English
                translated_value = english_value
            
            crop_details[key] = translated_value
        
        # Handle case where the prediction failed or details are missing
        if 'Unknown Crop' in predicted_crop_english:
             crop_details = {k: T['not_available'] for k in ['Sowing', 'Duration', 'Harvest']}


        # Custom CSS for the colored cards/boxes
        st.markdown("""
        <style>
        .prediction-box {
            background-color: #e0f7fa; 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            margin-bottom: 20px;
        }
        .time-card {
            background-color: #f1f8e9; 
            padding: 15px; 
            border-radius: 8px; 
            height: 100%;
            border-left: 5px solid #689f38;
        }
        .time-card h4 {
            color: #33691e;
            margin-top: 0;
            margin-bottom: 5px;
        }
        .time-card p {
            font-size: 16px;
            font-weight: 500;
            color: #212121;
        }
        </style>
        """, unsafe_allow_html=True)


        # 1. Main Recommendation Box
        st.markdown(f"""
            <div class="prediction-box">
                <h2>{T['rec_header']}</h2>
                <p style="font-size:36px;font-weight:700;color:#004d40;">{predicted_crop_display.upper()}</p>
                <p>{T['rec_base_text'].format(f'<b style="color:#008080;">{predicted_crop_display.upper()}</b>')}</p>
            </div>
        """, unsafe_allow_html=True)

        # 2. Time Details Section (Content is now translated)
        st.subheader(T['time_subheader'])
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="time-card">
                <h4>{T['sowing_time']}</h4>
                <p>{crop_details['Sowing']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="time-card">
                <h4>{T['duration_time']}</h4>
                <p>{crop_details['Duration']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="time-card">
                <h4>{T['harvest_time']}</h4>
            <p>{crop_details['Harvest']}</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(T['error_pred'].format(e))
        st.info(T['info_pred'])

st.markdown("---")
st.caption(T['note'])