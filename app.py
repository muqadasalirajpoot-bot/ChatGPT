import streamlit as st
from openai import OpenAI, AuthenticationError, APIConnectionError, APIStatusError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="ChatGPT + LangChain",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = "You are a chatbot having conversation with a human"

def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Validate the supplied key without generating a chat response."""
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, "API key is valid."
    except AuthenticationError:
        return False, "Invalid OpenAI API key. Please check the key and try again."
    except APIConnectionError:
        return False, "Could not connect to OpenAI. Check your internet connection and try again."
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            return False, "The API key was rejected or does not have permission to use the API."
        return False, f"OpenAI returned an API error ({exc.status_code})."
    except Exception as exc:
        return False, f"Unable to validate the API key: {exc}"

def build_chain(api_key: str, model_name: str, temperature: float):
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{content}"),
    ])

    return prompt | llm | StrOutputParser()

def reset_app():
    for key in ("api_key", "api_valid", "messages", "chain"):
        st.session_state.pop(key, None)

# ---------- First screen: API key gate ----------
if not st.session_state.get("api_valid", False):
    st.markdown(
        """
        <div style="text-align:center; padding: 70px 20px 25px;">
            <div style="font-size:72px;">🤖</div>
            <h1>ChatGPT + LangChain</h1>
            <p style="font-size:18px;">
                Enter your OpenAI API key to unlock the chatbot.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("🔐 API Key Required")
        st.info(
            "Your key is entered by you at runtime. This app does not place "
            "a hard-coded API key in the source code."
        )

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your own OpenAI API key.",
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            verify = st.button(
                "🔓 Verify API Key & Continue",
                type="primary",
                use_container_width=True,
                disabled=not api_key.strip(),
            )

        if verify:
            with st.spinner("Verifying your API key..."):
                valid, message = validate_api_key(api_key.strip())

            if valid:
                st.session_state.api_key = api_key.strip()
                st.session_state.api_valid = True
                st.session_state.messages = []
                st.success("✅ API key verified. Opening the chatbot...")
                st.rerun()
            else:
                st.error(f"❌ {message}")

    st.caption(
        "For security, never paste your API key into GitHub, source code, "
        ".env.example, screenshots, or public repositories."
    )
    st.stop()

# ---------- Chat screen ----------
with st.sidebar:
    st.title("⚙️ Chat Settings")
    st.success("🔐 API key verified")

    model_name = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4o-mini"],
        index=0,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔑 Change API Key", use_container_width=True):
        reset_app()
        st.rerun()

    st.divider()
    st.caption("Backend flow from the supplied notebook:")
    st.code("Prompt → ChatOpenAI → StrOutputParser", language="text")

st.title("🤖 ChatGPT + LangChain")
st.caption("A Streamlit frontend for your LangChain chatbot backend")

# Rebuild the chain if settings changed.
settings_signature = (model_name, temperature)
if st.session_state.get("chain_settings") != settings_signature:
    st.session_state.chain = build_chain(
        st.session_state.api_key,
        model_name,
        temperature,
    )
    st.session_state.chain_settings = settings_signature

# Welcome message
if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hello! 👋 Your API key has been verified. How can I help you today?"
        }
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.chain.invoke({"content": user_input})
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        except AuthenticationError:
            st.error("❌ OpenAI rejected the API key. Please use Change API Key and enter a valid key.")
            st.session_state.api_valid = False
        except Exception as exc:
            st.error(f"❌ Request failed: {exc}")

st.divider()
st.caption("Educational frontend • API key is supplied at runtime")
