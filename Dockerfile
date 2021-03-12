FROM python:3.8
# Create working directory
WORKDIR /nguyen-apps

# Copy requirements. From source, to destination.
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8080

# copying all files over. From source, to destination.
COPY . /nguyen-apps

#Run app
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
