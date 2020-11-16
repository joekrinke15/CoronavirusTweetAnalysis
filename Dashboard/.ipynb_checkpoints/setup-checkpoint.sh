mkdir -p ~/.streamlit/
echo “\
[general]\n\
email = \”jdk61@duke.edu”\n\
“ > ~/.streamlit/credentials.toml
echo “\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
“ > ~/.streamlit/config.toml
