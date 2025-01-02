# e-Prahari: Digital Sentinel Framework

A comprehensive framework for detecting and countering digital misinformation through an aggregated credibility scoring system.

## Overview

e-Prahari is an advanced misinformation detection system that combines multiple analysis techniques:

- Source credibility assessment
- Content consistency verification
- Engagement pattern analysis
- Author behavior tracking
- Deep learning-based content analysis
- Image manipulation detection

The system provides a comprehensive scoring methodology to evaluate digital content credibility and identify potential misinformation.

## Key Features

- **Multiple Analysis Layers**: Combines text, image, and metadata analysis
- **Comprehensive Scoring**: Aggregated credibility metrics with weighted importance
- **Scalable Architecture**: Five-layer design for real-time processing
- **Advanced ML Models**: Uses state-of-the-art deep learning for content analysis
- **API Interface**: RESTful API for easy integration
- **Detailed Reports**: In-depth analysis reports with actionable recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/e-prahari.git
cd e-prahari
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

```bash
python main.py serve
```

### Analyzing Content via CLI

```bash
python main.py analyze path/to/content.txt
```

### API Endpoints

- `POST /api/v1/analyze`: Analyze content for misinformation
  - Accepts text content and images
  - Returns comprehensive analysis results

## Configuration

Configure the system by setting environment variables or creating a `.env` file:

```env
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
FACT_CHECK_API_KEY=your-api-key
```

## Architecture

The system consists of five main layers:

1. **Data Collection**: Integrates multiple data sources
2. **Preprocessing**: Handles text and media normalization
3. **Processing**: Performs core analysis tasks
4. **Analysis**: Computes credibility scores
5. **Presentation**: Delivers results via API

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{hooda2024eprahari,
  title={e-Prahari: A Digital Sentinel Framework for Misinformation Detection Using Aggregated Credibility Metrics},
  author={Hooda, Saksham},
  year={2024}
}
```

## Contact

Saksham Hooda - sakshamhooda_mc20a7_62@dtu.ac.in