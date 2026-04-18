"""
app.py
------
Streamlit frontend for the Rail Madad grievance assistant.

Flow:
    1. Register — enter phone number, a user_id is created.
    2. Booking? — ask if the user has a ticket.
                    • Yes → upload the ticket, then go straight to chat.
                    • Skip → go straight to chat (no ticket on file).
    3. Chat     — Sarvam tool-using agent (journey lookup, grievance SMS,
                    RAG, web search).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────
# Base URL of the FastAPI backend (`api.py`). When deployed as a Databricks
# App this will be something like https://<app-name>.cloud.databricks.com.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
# Optional bearer token if the Databricks App is secured.
API_TOKEN = os.environ.get("API_TOKEN", "")


# ── Page setup & elder-friendly styling ──────────────────────────────────────

st.set_page_config(
    page_title="Rail Madad Sahayak",
    page_icon="🚆",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 640px; padding-top: 2.5rem; }
      .stButton > button { border-radius: 10px; width: 100%; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Localization (10 Indian languages) ────────────────────────────────────

# code → native display name used in the language picker
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "kn": "ಕನ್ನಡ",
    "ml": "മലയാളം",
    "pa": "ਪੰਜਾਬੀ",
}

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "mic_label": "🎤 Or speak your message",
        "transcribing": "Transcribing…",
        "transcribe_err": "Sorry, we could not understand the audio:",
        "app_title": "🚆 Rail Madad Sahayak",
        "register_caption": "Enter your phone number to begin.",
        "phone_placeholder": "10-digit mobile",
        "continue": "Continue",
        "invalid_phone": "Please enter a valid 10-digit phone number.",
        "backend_err": "Could not reach the backend:",
        "booking_title": "Do you have a booking?",
        "booking_caption": "Upload your ticket so we can help you faster, or skip to chat.",
        "yes_upload": "Yes, upload ticket",
        "skip": "Skip",
        "upload_title": "Upload your ticket",
        "upload_caption": "JPG, PNG or PDF. We'll read it for you.",
        "submit": "Submit",
        "reading_ticket": "Reading your ticket…",
        "ticket_read_err": "Sorry, we could not read the ticket:",
        "ticket_process_err": "Ticket could not be processed.",
        "chat_greeting": (
            "Namaste 🙏 Tell me what problem you are facing. "
            "You can write in Hindi, English, or any Indian language."
        ),
        "chat_placeholder": "Type your message…",
        "thinking": "Thinking…",
        "chat_unreachable": (
            "Sorry, the assistant is not reachable right now. "
            "Please try again in a moment."
        ),
        "no_response": "Sorry, I could not generate a response.",
        "send_sms": "📩 Send to Rail Madad (139)",
        "start_over": "Start over",
        "language": "Language",
    },
    "bn": {
        "mic_label": "🎤 অথবা আপনার বার্তা বলুন",
        "transcribing": "প্রতিলিপি করা হচ্ছে…",
        "transcribe_err": "দুঃখিত, আমরা অডিও বুঝতে পারিনি:",
        "app_title": "🚆 রেল মদদ সহায়ক",
        "register_caption": "শুরু করতে আপনার ফোন নম্বর লিখুন।",
        "phone_placeholder": "10-সংখ্যার মোবাইল",
        "continue": "এগিয়ে যান",
        "invalid_phone": "অনুগ্রহ করে একটি বৈধ 10-সংখ্যার ফোন নম্বর লিখুন।",
        "backend_err": "সার্ভারে পৌঁছানো যায়নি:",
        "booking_title": "আপনার কি বুকিং আছে?",
        "booking_caption": "আপনার টিকিট আপলোড করুন যাতে আমরা দ্রুত সাহায্য করতে পারি, বা নেরাসরি চ্যাটে চলে যান।",
        "yes_upload": "হ্যাঁ, টিকিট আপলোড করুন",
        "skip": "এড়িয়ে যান",
        "upload_title": "আপনার টিকিট আপলোড করুন",
        "upload_caption": "JPG, PNG বা PDF। আমরা এটি আপনার জন্য পড়ব।",
        "submit": "জমা দিন",
        "reading_ticket": "আপনার টিকিট পড়া হচ্ছে…",
        "ticket_read_err": "দুঃখিত, আমরা টিকিট পড়তে পারিনি:",
        "ticket_process_err": "টিকিট প্রক্রিয়া করা যায়নি।",
        "chat_greeting": (
            "নমস্কার 🙏 আপনি কী সমস্যায় পড়েছেন বলুন। "
            "আপনি যেকোনো ভারতীয় ভাষায় লিখতে পারেন।"
        ),
        "chat_placeholder": "আপনার বার্তা লিখুন…",
        "thinking": "ভাবছি…",
        "chat_unreachable": (
            "দুঃখিত, সহায়ক এখন উপলব্ধ নয়। "
            "অনুগ্রহ করে কিছুক্ষণ পরে আবার চেষ্টা করুন।"
        ),
        "no_response": "দুঃখিত, আমি উত্তর তৈরি করতে পারিনি।",
        "send_sms": "📩 রেল মদদ (139)-এ পাঠান",
        "start_over": "আবার শুরু করুন",
        "language": "ভাষা",
    },
    "ta": {
        "mic_label": "🎤 அல்லது உங்கள் செயதியை பேசவும்",
        "transcribing": "படியெடுக்கிறேன்…",
        "transcribe_err": "மன்னிக்கவும், ஒலியை புரிந்துக்கொள்ள முடியவில்லை:",
        "app_title": "🚆 ரயில் மதத் சஹாயக்",
        "register_caption": "தொடங்க உங்கள் தொலைபேசி எண்ணை உள்ளிடவும்.",
        "phone_placeholder": "10-இலக்க மொபைல்",
        "continue": "தொடரவும்",
        "invalid_phone": "தயவுசெய்து சரியான 10-இலக்க தொலைபேசி எண்ணை உள்ளிடவும்.",
        "backend_err": "சர்வரை அடைய முடியவில்லை:",
        "booking_title": "உங்களிடம் முன்பதிவு உள்ளதா?",
        "booking_caption": "உங்கள் டிக்கெட்டை பதிவேற்றவும், நாம் விரைவாக உதவ முடியும், அல்லது நேரடியாக அரட்டைக்குச் செல்லவும்.",
        "yes_upload": "ஆம், டிக்கெட்டை பதிவேற்றவும்",
        "skip": "தவிர்க்கவும்",
        "upload_title": "உங்கள் டிக்கெட்டை பதிவேற்றவும்",
        "upload_caption": "JPG, PNG அல்லது PDF. உங்களுக்காக நாம் படிப்போம்.",
        "submit": "சமர்ப்பிக்கவும்",
        "reading_ticket": "உங்கள் டிக்கெட்டை படிக்கிறேன்…",
        "ticket_read_err": "மன்னிக்கவும், டிக்கெட்டை படிக்க முடியவில்லை:",
        "ticket_process_err": "டிக்கெட்டைச் செயலாக்க முடியவில்லை.",
        "chat_greeting": (
            "வணக்கம் 🙏 உங்களுக்கு என்ன பிரச்சினை என்பதைச் சொல்லுங்கள். "
            "எந்த இந்திய மொழியிலும் எழுதலாம்."
        ),
        "chat_placeholder": "உங்கள் செய்தியை தட்டச்சு செய்யவும்…",
        "thinking": "சிந்திக்கிறேன்…",
        "chat_unreachable": (
            "மன்னிக்கவும், உதவியாளர் இப்போது கிடைக்கவில்லை. "
            "சிறிது நேரத்தில் மீண்டும் முயற்சிக்கவும்."
        ),
        "no_response": "மன்னிக்கவும், நான் பதில் உருவாக்க முடியவில்லை.",
        "send_sms": "📩 ரயில் மதத் (139)-க்கு அனுப்பவும்",
        "start_over": "மீண்டும் தொடங்கவும்",
        "language": "மொழி",
    },
    "te": {
        "mic_label": "🎤 లేదా మీ సందేశాన్ని మాట్లాడండి",
        "transcribing": "లిఖితం చేస్తున్నాము…",
        "transcribe_err": "క్షమించండి, ఆడియోను అర్థం చేసుకోలేకపోయాము:",
        "app_title": "🚆 రైల్ మదద్ సహాయక్",
        "register_caption": "ప్రారంభించడానికి మీ ఫోన్ నంబర్‌ను నమోదు చేయండి.",
        "phone_placeholder": "10-అంకెల మొబైల్",
        "continue": "కొనసాగించండి",
        "invalid_phone": "దయచేసి చెల్లుబాటు అయ్యే 10-అంకెల ఫోన్ నంబర్‌ను నమోదు చేయండి.",
        "backend_err": "సర్వర్‌ను చేరుకోలేకపోయాము:",
        "booking_title": "మీకు బుకింగ్ ఉందా?",
        "booking_caption": "మీ టికెట్‌ను అప్‌లోడ్ చేయండి, మేము త్వరగా సహాయం చేయగలం, లేదా నేరుగా చాట్‌కి వెళ్ళండి.",
        "yes_upload": "అవును, టికెట్ అప్‌లోడ్ చేయండి",
        "skip": "దాటవేయండి",
        "upload_title": "మీ టికెట్‌ను అప్‌లోడ్ చేయండి",
        "upload_caption": "JPG, PNG లేదా PDF. మేము మీ కొరకు దాన్ని చదువుతాము.",
        "submit": "సమర్పించండి",
        "reading_ticket": "మీ టికెట్‌ను చదువుతున్నాము…",
        "ticket_read_err": "క్షమించండి, టికెట్‌ను చదవలేకపోయాము:",
        "ticket_process_err": "టికెట్‌ను ప్రాసెస్ చేయలేకపోయాము.",
        "chat_greeting": (
            "నమస్కారం 🙏 మీకు ఏ సమస్య ఉందో చెప్పండి. "
            "మీరు ఏ భారతీయ భాషలోనైనా వ్రాయవచ్చు."
        ),
        "chat_placeholder": "మీ సందేశాన్ని టైప్ చేయండి…",
        "thinking": "ఆలోచిస్తున్నాను…",
        "chat_unreachable": (
            "క్షమించండి, సహాయకుడు ప్రస్తుతం అందుబాటులో లేరు. "
            "దయచేసి కొంతసేపటి తర్వాత మళ్లీ ప్రయత్నించండి."
        ),
        "no_response": "క్షమించండి, నేను సమాధానం రూపొందించలేకపోయాను.",
        "send_sms": "📩 రైల్ మదద్ (139)కి పంపండి",
        "start_over": "మళ్లీ ప్రారంభించండి",
        "language": "భాష",
    },
    "mr": {
        "mic_label": "🎤 किंवा आपला संदेश बोला",
        "transcribing": "लिप्यंतर करत आहे…",
        "transcribe_err": "क्षमस्व, आम्ही ऑडिओ समजू शकलो नाही:",
        "app_title": "🚆 रेल मदत सहायक",
        "register_caption": "सुरू करण्यासाठी आपला फोन नंबर प्रविष्ट करा.",
        "phone_placeholder": "10-अंकी मोबाइल",
        "continue": "पुढे जा",
        "invalid_phone": "कृपया वैध 10-अंकी फोन नंबर प्रविष्ट करा.",
        "backend_err": "सर्व्हरशी संपर्क झाला नाही:",
        "booking_title": "आपल्याकडे बुकिंग आहे का?",
        "booking_caption": "आपले तिकीट अपलोड करा जेणेकरून आम्ही आपल्याला जलद मदत करू शकू, किंवा थेट गप्पांवर जा.",
        "yes_upload": "होय, तिकीट अपलोड करा",
        "skip": "वगळा",
        "upload_title": "आपले तिकीट अपलोड करा",
        "upload_caption": "JPG, PNG किंवा PDF. आम्ही आपल्यासाठी ते वाचू.",
        "submit": "सबमिट करा",
        "reading_ticket": "आपले तिकीट वाचत आहे…",
        "ticket_read_err": "क्षमस्व, आम्ही तिकीट वाचू शकलो नाही:",
        "ticket_process_err": "तिकीट प्रक्रिया करता आले नाही.",
        "chat_greeting": (
            "नमस्कार 🙏 आपल्याला कोणती अडचण आहे ते सांगा. "
            "आपण कोणत्याही भारतीय भाषेत लिहू शकता."
        ),
        "chat_placeholder": "आपला संदेश टाइप करा…",
        "thinking": "विचार करत आहे…",
        "chat_unreachable": (
            "क्षमस्व, सहायक सध्या उपलब्ध नाही. "
            "कृपया काही वेळाने पुन्हा प्रयत्न करा."
        ),
        "no_response": "क्षमस्व, मी उत्तर तयार करू शकलो नाही.",
        "send_sms": "📩 रेल मदत (139) कडे पाठवा",
        "start_over": "पुन्हा सुरू करा",
        "language": "भाषा",
    },
    "gu": {
        "mic_label": "🎤 અથવા તમારો સંદેશ બોલો",
        "transcribing": "લખી રહ્યા છીએ…",
        "transcribe_err": "માફ કરશો, અમે ઑડિઓ સમજી શક્યા નહીં:",
        "app_title": "🚆 રેલ મદદ સહાયક",
        "register_caption": "શરૂ કરવા માટે તમારો ફોન નંબર દાખલ કરો.",
        "phone_placeholder": "10-અંકી મોબાઇલ",
        "continue": "આગળ વધો",
        "invalid_phone": "કૃપા કરી માન્ય 10-અંકી ફોન નંબર દાખલ કરો.",
        "backend_err": "સર્વર સાથે સંપર્ક થયો નહીં:",
        "booking_title": "શું તમારી પાસે બુકિંગ છે?",
        "booking_caption": "તમારી ટિકિટ અપલોડ કરો જેથી અમે તમને ઝડપથી મદદ કરી શકીએ, અથવા સીધા ચેટ પર જાઓ.",
        "yes_upload": "હા, ટિકિટ અપલોડ કરો",
        "skip": "છોડો",
        "upload_title": "તમારી ટિકિટ અપલોડ કરો",
        "upload_caption": "JPG, PNG અથવા PDF. અમે તમારા માટે તે વાંચીશું.",
        "submit": "સબમિટ કરો",
        "reading_ticket": "તમારી ટિકિટ વાંચી રહ્યા છીએ…",
        "ticket_read_err": "માફ કરશો, અમે ટિકિટ વાંચી શક્યા નહીં:",
        "ticket_process_err": "ટિકિટ પ્રક્રિયા થઈ શકી નહીં.",
        "chat_greeting": (
            "નમસ્કાર 🙏 તમને કઈ તકલીફ છે તે કહો. "
            "તમે કોઈપણ ભારતીય ભાષામાં લખી શકો છો."
        ),
        "chat_placeholder": "તમારો સંદેશ લખો…",
        "thinking": "વિચારી રહ્યા છીએ…",
        "chat_unreachable": (
            "માફ કરશો, સહાયક હાલમાં ઉપલબ્ધ નથી. "
            "થોડી વાર પછી ફરી પ્રયાસ કરો."
        ),
        "no_response": "માફ કરશો, હું જવાબ ત્યાર કરી શક્યો નહીં.",
        "send_sms": "📩 રેલ મદદ (139) પર મોકલો",
        "start_over": "ફરી શરૂ કરો",
        "language": "ભાષા",
    },
    "kn": {
        "mic_label": "🎤 ಅಥವಾ ನಿಮ್ಮ ಸಂದೇಶವನ್ನು ಮಾತನಾಡಿ",
        "transcribing": "ಪ್ರತಿಲಿಪಿಗೊಳಿಸಲಾಗುತ್ತಿದೆ…",
        "transcribe_err": "ಕ್ಷಮಿಸಿ, ನಾವು ಆಡಿಯೋ ಅರ್ಥಮಾಡಿಕೋಳ್ಳಲಾಗಲಿಲ್ಲ:",
        "app_title": "🚆 ರಈಲ್ ಮದದ್ ಸಹಾಯಕ",
        "register_caption": "ಪ್ರಾರಂಭಿಸಲು ನಿಮ್ಮ ಫೋನ್ ಸಂಖ್ಯೆಯನ್ನು ನಮೂದಿಸಿ.",
        "phone_placeholder": "10-ಅಂಕಿಯ ಮೊಬೈಲ್",
        "continue": "ಮುಂದುವರಿಸಿ",
        "invalid_phone": "ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ 10-ಅಂಕಿಯ ಫೋನ್ ಸಂಖ್ಯೆಯನ್ನು ನಮೂದಿಸಿ.",
        "backend_err": "ಸರ್ವರ್ ಅನ್ನು ತಲುಪಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ:",
        "booking_title": "ನಿಮ್ಮ ಬಳಿ ಬುಕಿಂಗ್ ಇದೆಯೇ?",
        "booking_caption": "ನಿಮ್ಮ ಟಿಕೆಟ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ, ನಾವು ಬೇಗ ಸಹಾಯ ಮಾಡಬಹುದು, ಅಥವಾ ನೇರವಾಗಿ ಚಾಟ್‌ಗೆ ಹೋಗಿ.",
        "yes_upload": "ಹೌದು, ಟಿಕೆಟ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "skip": "ಬಿಟ್ಟುಬಿಡಿ",
        "upload_title": "ನಿಮ್ಮ ಟಿಕೆಟ್ ಅನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "upload_caption": "JPG, PNG ಅಥವಾ PDF. ನಾವು ನಿಮಗಾಗಿ ಓದುತ್ತೇವೆ.",
        "submit": "ಸಲ್ಲಿಸಿ",
        "reading_ticket": "ನಿಮ್ಮ ಟಿಕೆಟ್ ಓದುತ್ತಿದ್ದೇವೆ…",
        "ticket_read_err": "ಕ್ಷಮಿಸಿ, ನಾವು ಟಿಕೆಟ್ ಓದಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ:",
        "ticket_process_err": "ಟಿಕೆಟ್ ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
        "chat_greeting": (
            "ನಮಸ್ಕಾರ 🙏 ನಿಮಗೆ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ಹೇಳಿ. "
            "ನೀವು ಯಾವುದೇ ಭಾರತೀಯ ಭಾಷೆಯಲ್ಲಿ ಬರೆಯಬಹುದು."
        ),
        "chat_placeholder": "ನಿಮ್ಮ ಸಂದೇಶವನ್ನು ಟೈಪ್ ಮಾಡಿ…",
        "thinking": "ಯೋಚಿಸುತ್ತಿದ್ದೇನೆ…",
        "chat_unreachable": (
            "ಕ್ಷಮಿಸಿ, ಸಹಾಯಕ ಸದ್ಯಕ್ಕೆ ಲಭ್ಯವಿಲ್ಲ. "
            "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಪುನರ3 ಪ್ರಯತ್ನಿಸಿ."
        ),
        "no_response": "ಕ್ಷಮಿಸಿ, ನಾನು ಉತ್ತರ ರೂಪಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
        "send_sms": "📩 ರಈಲ್ ಮದದ್ (139) ಗೆ ಕಳುಹಿಸಿ",
        "start_over": "ಮತ್ತೆ ಪ್ರಾರಂಭಿಸಿ",
        "language": "ಭಾಷೆ",
    },
    "ml": {
        "mic_label": "🎤 അല്ലെങ്കിൽ നിങ്ങളുടെ സന്ദേശം പറയൂ",
        "transcribing": "ട്രാൻസ്‌ക്രൈബ് ചെയ്യുന്നു…",
        "transcribe_err": "ക്ഷമിക്കണം, ഒാഡിയോ മനസ്സിലാക്കാനായില്ല:",
        "app_title": "🚆 റെയിൽ മദദ് സഹായക്",
        "register_caption": "തുടങ്ങാൻ നിങ്ങളുടെ ഫോൺ നമ്പർ നൽകുക.",
        "phone_placeholder": "10-അക്ക മൊബൈൽ",
        "continue": "തുടരുക",
        "invalid_phone": "ദയവായി സാധുവായ 10-അക്ക ഫോൺ നമ്പർ നൽകുക.",
        "backend_err": "സർവറുമായി ബന്ധപ്പെടാനായില്ല:",
        "booking_title": "നിങ്ങൾക്ക് ബുക്കിങ് ഉണ്ടോ?",
        "booking_caption": "നിങ്ങളുടെ ടിക്കറ്റ് അപ്‌ലോഡ് ചെയ്യൂ, ഇതുവഴി ഞങ്ങൾക്ക് വേഗത്തിൽ സഹായിക്കാം, അല്ലെങ്കിൽ നേരിട്ട് ചാറ്റിലേക്ക് പോകൂ.",
        "yes_upload": "അതെ, ടിക്കറ്റ് അപ്‌ലോഡ് ചെയ്യുക",
        "skip": "ഒഴിവാക്കുക",
        "upload_title": "നിങ്ങളുടെ ടിക്കറ്റ് അപ്‌ലോഡ് ചെയ്യുക",
        "upload_caption": "JPG, PNG അല്ലെങ്കിൽ PDF. ഞങ്ങൾ നിങ്ങൾക്കായി വായിക്കാം.",
        "submit": "സമർപ്പിക്കുക",
        "reading_ticket": "നിങ്ങളുടെ ടിക്കറ്റ് വായിക്കുന്നു…",
        "ticket_read_err": "ക്ഷമിക്കണം, ഞങ്ങൾക്ക് ടിക്കറ്റ് വായിക്കാനായില്ല:",
        "ticket_process_err": "ടിക്കറ്റ് പ്രോസസ് ചെയ്യാനായില്ല.",
        "chat_greeting": (
            "നമസ്കാരം 🙏 നിങ്ങൾക്ക് എന്ത് പ്രശ്‌നമാണെന്ന് പറയൂ. "
            "ഏത് ഇന്ത്യൻ ഭാഷയിലും എഴുതാം."
        ),
        "chat_placeholder": "നിങ്ങളുടെ സന്ദേശം ടൈപ്പ് ചെയ്യുക…",
        "thinking": "ആലോചിക്കുന്നു…",
        "chat_unreachable": (
            "ക്ഷമിക്കണം, സഹായി ഇപ്പോൾ ലഭ്യമല്ല. "
            "ദയവായി കുറച്ചു കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക."
        ),
        "no_response": "ക്ഷമിക്കണം, എനിക്ക് ഉത്തരം സൃഷ്ടിക്കാനായില്ല.",
        "send_sms": "📩 റെയിൽ മദദ് (139)-ലേക്ക് അയയ്ക്കുക",
        "start_over": "വീണ്ടും ആരംഭിക്കുക",
        "language": "ഭാഷ",
    },
    "pa": {
        "mic_label": "🎤 ਜਾਂ ਆਪਣਾ ਸੁਨੇਹਾ ਬੋਲੋ",
        "transcribing": "ਲਿਪੀਅੰਤਰਣ ਹੋ ਰਿਹਾ ਹੈ…",
        "transcribe_err": "ਮਾਫ਼਼ ਕਰਨਾ, ਅਸੀਂ ਆਡੀਓ ਸਮਝ ਨਹੀਂ ਸਕੇ:",
        "app_title": "🚆 ਰੇਲ ਮਦਦ ਸਹਾਇਕ",
        "register_caption": "ਸ਼ੁਰੂ ਕਰਨ ਲਈ ਆਪਣਾ ਫ਼਼ੋਨ ਨੰਬਰ ਦਰਜ ਕਰੋ।",
        "phone_placeholder": "10-ਅੰਕਾਂ ਦਾ ਮੋਬਾਇਲ",
        "continue": "ਅੱਗੇ ਵਧੋ",
        "invalid_phone": "ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਵੈਧ 10-ਅੰਕਾਂ ਦਾ ਫ਼਼ੋਨ ਨੰਬਰ ਦਰਜ ਕਰੋ।",
        "backend_err": "ਸਰਵਰ ਨਾਲ ਸੰਪਰਕ ਨਹੀਂ ਹੋ ਸਕਿਆ:",
        "booking_title": "ਕੀ ਤੁਹਾਡੇ ਕੋਲ ਬੁਕਿੰਗ ਹੈ?",
        "booking_caption": "ਆਪਣੀ ਟਿਕਟ ਅਪਲੋਡ ਕਰੋ ਤਾਂ ਜੋ ਅਸੀਂ ਤੁਹਾਡੀ ਤੇਜ਼ੀ ਨਾਲ ਮਦਦ ਕਰ ਸਕੀਏ, ਜਾਂ ਸਿੱਧਾ ਚੈਟ 'ਤੇ ਜਾਓ।",
        "yes_upload": "ਹਾਂ, ਟਿਕਟ ਅਪਲੋਡ ਕਰੋ",
        "skip": "ਛੱਡੋ",
        "upload_title": "ਆਪਣੀ ਟਿਕਟ ਅਪਲੋਡ ਕਰੋ",
        "upload_caption": "JPG, PNG ਜਾਂ PDF। ਅਸੀਂ ਤੁਹਾਡੇ ਲਈ ਇਸਨੂੰ ਪੜ੍ਹਾਂਗੇ।",
        "submit": "ਜਮ੍ਹਾਂ ਕਰੋ",
        "reading_ticket": "ਤੁਹਾਡੀ ਟਿਕਟ ਪੜ੍ਹੀ ਜਾ ਰਹੀ ਹੈ…",
        "ticket_read_err": "ਮਾਫ਼਼ ਕਰਨਾ, ਅਸੀਂ ਟਿਕਟ ਨਹੀਂ ਪੜ੍ਹ ਸਕੇ:",
        "ticket_process_err": "ਟਿਕਟ ਪ੍ਰੋਸੈਸ ਨਹੀਂ ਹੋ ਸਕੀ।",
        "chat_greeting": (
            "ਸਤ ਸ੍ਰੀ ਅਕਾਲ 🙏 ਮੈਨੂੰ ਦੱਸੋ ਤੁਹਾਨੂੰ ਕੀ ਸਮੱਸਿਆ ਆ ਰਹੀ ਹੈ। "
            "ਤੁਸੀਂ ਕਿਸੇ ਵੀ ਭਾਰਤੀ ਭਾਸ਼ਾ ਵਿੱਚ ਲਿਖ ਸਕਦੇ ਹੋ।"
        ),
        "chat_placeholder": "ਆਪਣਾ ਸੁਨੇਹਾ ਲਿਖੋ…",
        "thinking": "ਸੋਚ ਰਿਹਾ ਹਾਂ…",
        "chat_unreachable": (
            "ਮਾਫ਼਼ ਕਰਨਾ, ਸਹਾਇਕ ਇਸ ਵੇਲੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। "
            "ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।"
        ),
        "no_response": "ਮਾਫ਼਼ ਕਰਨਾ, ਮੈਂ ਜਵਾਬ ਨਹੀਂ ਬਣਾ ਸਕਿਆ।",
        "send_sms": "📩 ਰੇਲ ਮਦਦ (139) ਨੂੰ ਭੇਜੋ",
        "start_over": "ਦੁਬਾਰਾ ਸ਼ੁਰੂ ਕਰੋ",
        "language": "ਭਾਸ਼ਾ",
    },
    "hi": {
        "mic_label": "🎤 या अपना संदेश बोलें",
        "transcribing": "सुना जा रहा है…",
        "transcribe_err": "क्षमा करें, अाडियो समझ नहीं आई:",
        "app_title": "🚆 रेल मदद सहायक",
        "register_caption": "शुरू करने के लिए अपना फ़ोन नंबर दर्ज करें।",
        "phone_placeholder": "10-अंकों का मोबाइल नंबर",
        "continue": "आगे बढ़ें",
        "invalid_phone": "कृपया एक वैध 10-अंकों का फ़ोन नंबर दर्ज करें।",
        "backend_err": "सर्वर से संपर्क नहीं हो पाया:",
        "booking_title": "क्या आपके पास बुकिंग है?",
        "booking_caption": "अपना टिकट अपलोड करें ताकि हम जल्दी मदद कर सकें, या सीधे चैट पर जाएँ।",
        "yes_upload": "हाँ, टिकट अपलोड करें",
        "skip": "छोड़ें",
        "upload_title": "अपना टिकट अपलोड करें",
        "upload_caption": "JPG, PNG या PDF। हम आपके लिए इसे पढ़ लेंगे।",
        "submit": "जमा करें",
        "reading_ticket": "आपका टिकट पढ़ा जा रहा है…",
        "ticket_read_err": "क्षमा करें, हम टिकट नहीं पढ़ पाए:",
        "ticket_process_err": "टिकट संसाधित नहीं किया जा सका।",
        "chat_greeting": (
            "नमस्ते 🙏 बताइए आपको क्या परेशानी है। "
            "आप हिंदी, अंग्रेज़ी या किसी भी भारतीय भाषा में लिख सकते हैं।"
        ),
        "chat_placeholder": "अपना संदेश लिखें…",
        "thinking": "सोच रहा हूँ…",
        "chat_unreachable": (
            "क्षमा करें, सहायक अभी उपलब्ध नहीं है। "
            "कृपया थोड़ी देर बाद पुनः प्रयास करें।"
        ),
        "no_response": "क्षमा करें, मैं उत्तर नहीं दे पाया।",
        "send_sms": "📩 रेल मदद (139) को भेजें",
        "start_over": "फिर से शुरू करें",
        "language": "भाषा",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return STRINGS.get(lang, STRINGS["en"]).get(key, STRINGS["en"].get(key, key))


# ── Session state defaults ───────────────────────────────────────────────────

def _init_state() -> None:
    # step: register | booking | ticket | chat
    st.session_state.setdefault("step", "register")
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("phone", "")
    st.session_state.setdefault("journey", None)          # dict of ticket fields
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("lang", "en")             # "en" | "hi"


_init_state()


# ── FastAPI backend helpers ───────────────────────────────────────────────────

def _auth_headers() -> dict:
    if API_TOKEN:
        return {"Authorization": f"Bearer {API_TOKEN}"}
    return {}


def call_register(phone: str) -> dict:
    """POST /users/register — returns {user_id, phone, status, message}."""
    resp = requests.post(
        f"{API_BASE_URL}/users/register",
        json={"phone": phone},
        headers=_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_ticket_upload(image_bytes: bytes, filename: str, user_id: str) -> dict:
    """POST /tickets/upload — multipart, returns {journey_id, extracted, ...}."""
    resp = requests.post(
        f"{API_BASE_URL}/tickets/upload",
        files={"file": (filename, image_bytes)},
        data={"user_id": user_id},
        headers=_auth_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def call_chat(query: str, user_id: str, lang: str = "en",
              history: list[dict] | None = None) -> dict:
    """POST /chat — returns {reply, sms_uri, tool_log}."""
    resp = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "query": query,
            "user_id": user_id,
            "lang": lang,
            "history": history or [],
        },
        headers=_auth_headers(),
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def call_transcribe(audio_bytes: bytes, filename: str, content_type: str,
                    lang: str) -> dict:
    """POST /speech/transcribe — returns {text, source_language_code}."""
    resp = requests.post(
        f"{API_BASE_URL}/speech/transcribe",
        files={"file": (filename, audio_bytes, content_type or "audio/wav")},
        data={"lang": lang},
        headers=_auth_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ── Shared UI helpers ────────────────────────────────────────────────────────

def _render_sms_link(sms_uri: str) -> None:
    """Render the Rail Madad grievance SMS URI as a clickable link."""
    if not sms_uri:
        return
    st.markdown(
        f'<a href="{sms_uri}" style="display:inline-block;margin-top:8px;'
        'padding:10px 16px;background:#0B8043;color:#fff;border-radius:8px;'
        f'text-decoration:none;font-weight:600;">{t("send_sms")}</a>',
        unsafe_allow_html=True,
    )


# ── Step 1 — Registration ────────────────────────────────────────────────────

def render_register() -> None:
    st.title(t("app_title"))
    st.caption(t("register_caption"))

    phone = st.text_input(
        "phone",
        max_chars=10,
        placeholder=t("phone_placeholder"),
        key="phone_input",
        label_visibility="collapsed",
    )

    if st.button(t("continue"), type="primary"):
        if not (phone or "").isdigit() or len(phone) != 10:
            st.error(t("invalid_phone"))
            return
        try:
            result = call_register(phone)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{t('backend_err')} {exc}")
            return
        st.session_state.user_id = result.get("user_id", "")
        st.session_state.phone = phone
        st.session_state.step = "booking"
        st.rerun()


# ── Step 2 — Do you have a booking? ──────────────────────────────────────────

def render_booking() -> None:
    st.title(t("booking_title"))
    st.caption(t("booking_caption"))

    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("yes_upload"), type="primary"):
            st.session_state.step = "ticket"
            st.rerun()
    with col2:
        if st.button(t("skip")):
            st.session_state.step = "chat"
            st.rerun()


# ── Step 3 — Ticket upload ───────────────────────────────────────────────────

def render_ticket() -> None:
    st.title(t("upload_title"))
    st.caption(t("upload_caption"))

    uploaded = st.file_uploader(
        "ticket",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=False,
        key="ticket_uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        if uploaded.type and uploaded.type.startswith("image/"):
            st.image(uploaded, use_column_width=True)

        if st.button(t("submit"), type="primary"):
            with st.spinner(t("reading_ticket")):
                try:
                    result = call_ticket_upload(
                        image_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        user_id=st.session_state.user_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{t('ticket_read_err')} {exc}")
                    return

            if str(result.get("status", "")).lower() != "success":
                st.error(
                    f"{t('ticket_process_err')} {result.get('message', '')}"
                )
                return

            extracted = result.get("extracted") or {}
            st.session_state.journey = {
                "journey_id": result.get("journey_id", ""),
                **extracted,
            }
            # Go straight to chat after successful booking upload.
            st.session_state.step = "chat"
            st.rerun()

    if st.button(t("skip"), key="ticket_skip"):
        st.session_state.step = "chat"
        st.rerun()


# ── Step 4 — Chat ────────────────────────────────────────────────────────────

def _handle_user_prompt(prompt: str) -> None:
    """Push a user message through the chat pipeline and render the reply."""
    # Snapshot the history BEFORE appending the new user prompt so the backend
    # receives prior turns as context, and the current turn as `query`.
    history = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in st.session_state.chat_messages
        if m.get("role") in ("user", "assistant")
    ]
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t("thinking")):
            try:
                result = call_chat(
                    query=prompt,
                    user_id=st.session_state.user_id,
                    lang=st.session_state.get("lang", "en"),
                    history=history,
                )
            except Exception as exc:  # noqa: BLE001
                reply = f"{t('chat_unreachable')}\n\n_Error: {exc}_"
                st.markdown(reply)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": reply}
                )
                return

        reply = result.get("reply") or t("no_response")
        st.markdown(reply)

        sms_uri = result.get("sms_uri") or ""
        if sms_uri:
            _render_sms_link(sms_uri)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": reply, "sms_uri": sms_uri}
        )


def render_chat() -> None:
    j = st.session_state.journey or {}
    if j.get("pnr"):
        st.caption(
            f"PNR {j.get('pnr')} · {j.get('source_station', '')} → "
            f"{j.get('destination_station', '')}".strip(" ·")
        )

    if not st.session_state.chat_messages:
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": t("chat_greeting")}
        )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sms_uri"):
                _render_sms_link(msg["sms_uri"])

    # 🎤 Voice input (Streamlit ≥1.31). Records a clip in-browser and sends
    # it to /speech/transcribe, which returns text in the selected UI language.
    audio = st.audio_input(t("mic_label"), key="mic_input")
    if audio is not None:
        # Use the UploadedFile's stable `file_id` (or a hash fallback) so the
        # same recording is not transcribed & replayed on every rerun.
        audio_bytes = audio.getvalue() or b""
        clip_id = getattr(audio, "file_id", None)
        if not clip_id and audio_bytes:
            import hashlib  # local import — only needed for fallback
            clip_id = hashlib.sha1(audio_bytes).hexdigest()
        if clip_id and clip_id != st.session_state.get("_last_mic_id"):
            st.session_state._last_mic_id = clip_id
            if audio_bytes:
                with st.spinner(t("transcribing")):
                    try:
                        tr = call_transcribe(
                            audio_bytes=audio_bytes,
                            filename=getattr(audio, "name", "audio.wav"),
                            content_type=getattr(audio, "type", "audio/wav"),
                            lang=st.session_state.get("lang", "en"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"{t('transcribe_err')} {exc}")
                        tr = None
                if tr and tr.get("text"):
                    _handle_user_prompt(tr["text"])
                    st.rerun()

    prompt = st.chat_input(t("chat_placeholder"))
    if prompt:
        _handle_user_prompt(prompt)


# ── Sidebar (minimal: reset only) ────────────────────────────────────────────

with st.sidebar:
    _lang_codes = list(LANGUAGE_NAMES.keys())
    _current = st.session_state.get("lang", "en")
    lang_choice = st.selectbox(
        t("language"),
        options=_lang_codes,
        format_func=lambda c: LANGUAGE_NAMES.get(c, c),
        index=_lang_codes.index(_current) if _current in _lang_codes else 0,
        key="lang_selector",
    )
    if lang_choice != st.session_state.get("lang"):
        st.session_state.lang = lang_choice
        st.rerun()

    if st.session_state.phone:
        st.caption(f"📞 {st.session_state.phone}")
    if st.button(t("start_over")):
        lang = st.session_state.get("lang", "en")
        for k in ("step", "user_id", "phone", "journey", "chat_messages"):
            st.session_state.pop(k, None)
        _init_state()
        st.session_state.lang = lang
        st.rerun()


# ── Router ───────────────────────────────────────────────────────────────────

step = st.session_state.step
if step == "register":
    render_register()
elif step == "booking":
    render_booking()
elif step == "ticket":
    render_ticket()
elif step == "chat":
    render_chat()
else:
    st.session_state.step = "register"
    st.rerun()
