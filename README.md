# Company Research Tool

This project is a comprehensive company research tool that gathers and analyzes data from various sources to provide detailed insights about a company.

## Features

- Deep crawling of company websites
- Social media analysis
- News and updates aggregation
- Industry and competitor analysis
- Financial information gathering
- Technical stack analysis
- Employee reviews and company culture insights
- Cross-validation with additional sources

## Setup

### Prerequisites

- Python 3.7 or higher
- An internet connection for web crawling and API access

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/company-research-tool.git
   cd company-research-tool
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

- **API Key**: The project uses the Google GenAI API. You need to replace the `api_key` in `app.py` with your own API key.

### Usage

1. **Run the application:**

   ```bash
   python app.py
   ```

2. **Enter the company name** when prompted to start the research process.

3. **Results** will be saved in the `results` directory, organized by company name and date.

### Project Structure

- `app.py`: Main application file containing the logic for data gathering and analysis.
- `requirements.txt`: List of dependencies required to run the project.
- `results/`: Directory where research results are saved.

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

### License

This project is licensed under the MIT License.
