# CaseReviewer Python Server

A FastAPI-based server for social worker case review searches with PostgreSQL integration and AI-powered insights.

## Features

- **Vector Similarity Search**: Uses pgvector embeddings for semantic search across case reviews
- **AI-Powered Insights**: Generates personalized recommendations and risk assessments
- **Timeline Analysis**: Extracts and analyzes case timeline events
- **Multi-Agency Coordination**: Tracks involvement of different agencies
- **Personalized Summaries**: Creates role-specific insights for social workers
- **JWT Authentication**: Secure user authentication and authorization
- **PostgreSQL Integration**: Full database integration with existing schema

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database with pgvector extension
- Environment variables configured

## Installation

1. **Clone the repository and navigate to the server directory:**
   ```bash
   cd server
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the server directory:
   ```env
   DATABASE_URL=postgresql://username:password@host:port/database
   JWT_SECRET=your-secret-key-here
   PORT=5000
   ```

## Quick Start

### Option 1: Use the startup script (Recommended)
```bash
python start_server.py
```

### Option 2: Manual startup
```bash
python main.py
```

### Option 3: Use uvicorn directly
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

The server will start on `http://localhost:5000` (or the port specified in your environment).

## API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login

### Protected Routes (Require JWT token)
- `POST /api/protected/search` - Search case reviews with AI insights
- `GET /api/protected/case-reviews/{id}` - Get detailed case review
- `GET /api/protected/search-history` - Get user's search history
- `GET /api/protected/me` - Get current user information

### Health Check
- `GET /health` - Server health status

## API Usage Examples

### 1. User Registration
```bash
curl -X POST "http://localhost:5000/api/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "socialworker1",
    "password": "securepassword",
    "name": "John Doe",
    "role": "social_worker",
    "organization": "Local Authority"
  }'
```

### 2. User Login
```bash
curl -X POST "http://localhost:5000/api/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "socialworker1",
    "password": "securepassword"
  }'
```

### 3. Search Case Reviews
```bash
curl -X POST "http://localhost:5000/api/protected/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "missed opportunities in child protection",
    "filters": {
      "childAge": "0-5 years",
      "riskType": "neglect"
    },
    "top_k": 10
  }'
```

### 4. Get Case Review Details
```bash
curl -X GET "http://localhost:5000/api/protected/case-reviews/CASE_ID" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## AI Insights Features

### Personalized Insights
The server generates personalized insights for each search query, including:
- **Relevance Explanation**: Why the case matches the search
- **Key Lessons**: Important takeaways from the case
- **Actionable Recommendations**: Specific steps to take
- **Risk Assessment**: Analysis of risk levels and urgency
- **Intervention Strategies**: Evidence-based intervention approaches

### Case Complexity Analysis
- Complexity scoring based on multiple factors
- Risk type analysis
- Multi-agency involvement assessment
- Timeline complexity evaluation

### Timeline Insights
- Critical event identification
- Intervention point analysis
- Response time assessment
- Coordination event tracking

## Database Schema

The server works with the existing PostgreSQL schema that includes:

- `case_reviews` - Main case review data with embeddings
- `timeline_events` - Chronological case events
- `users` - User accounts and authentication
- `searches` - Search history tracking

## Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: Secret key for JWT tokens
- `PORT`: Server port (default: 5000)

### Database Requirements
- PostgreSQL 12 or higher
- pgvector extension enabled
- Tables created according to the existing schema

## Development

### Running Tests
```bash
pytest
```

### Code Structure
```
server/
├── main.py              # Main FastAPI application
├── services/
│   └── ai_insights.py  # AI insights service
├── requirements.txt     # Python dependencies
├── start_server.py     # Startup script
└── README.md           # This file
```

### Adding New Features
1. Create new service classes in the `services/` directory
2. Add new endpoints in `main.py`
3. Update Pydantic models as needed
4. Add tests for new functionality

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check `DATABASE_URL` environment variable
   - Ensure PostgreSQL is running
   - Verify pgvector extension is installed

2. **Dependencies Installation Failed**
   - Ensure Python 3.8+ is installed
   - Try installing dependencies manually: `pip install -r requirements.txt`

3. **Embedding Model Loading Failed**
   - The server will fall back to hash-based embeddings
   - Check internet connection for model download
   - Verify sufficient disk space

4. **Port Already in Use**
   - Change the `PORT` environment variable
   - Kill existing processes using the port

### Logs
The server provides detailed logging. Check the console output for:
- Database connection status
- Embedding model loading
- API request/response details
- Error messages and stack traces

## Performance Considerations

- **Embedding Model**: Uses lightweight `all-MiniLM-L6-v2` for fast processing
- **Database Indexes**: Ensure proper indexes on embedding and metadata fields
- **Caching**: Consider implementing Redis for frequently accessed data
- **Connection Pooling**: Database connections are managed efficiently

## Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Protected API endpoints
- Input validation with Pydantic
- CORS configuration for frontend integration

## Integration with Frontend

The server is designed to work with the existing React frontend. Key integration points:

- API endpoints match the expected structure
- JWT authentication flow
- Search results include all required fields
- Timeline events are properly formatted

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Verify environment configuration
4. Ensure database schema matches expectations

## License

This project is part of the CaseReviewer application for social worker case review searches.
