# Database Schema Setup for NSPCC Case Review System

This directory contains the database schema setup script that creates all necessary tables for the PDF processing pipeline.

## Files

- `setup_database_schema.py` - Main schema setup script
- `requirements_schema.txt` - Python dependencies for schema setup
- `README_SCHEMA_SETUP.md` - This documentation file

## Prerequisites

1. **PostgreSQL Database**: You need a PostgreSQL database running (local or cloud-based like Neon)
2. **Python Environment**: Python 3.7+ with pip
3. **Environment Variables**: Set up your `.env` file with database connection details

## Setup Instructions

### 1. Install Dependencies

```bash
cd scripts
pip install -r requirements_schema.txt
```

### 2. Configure Environment Variables

Create a `.env` file in your project root with:

```env
DATABASE_URL=postgresql://username:password@host:port/database
```

**Example for Neon:**
```env
DATABASE_URL=postgresql://username:password@ep-cool-name.us-east-1.aws.neon.tech/database?sslmode=require
```

### 3. Run the Schema Setup

```bash
python setup_database_schema.py
```

Choose option **1** to set up the complete schema.

## What Gets Created

### `case_reviews` Table
- **Core fields**: `id`, `title`, `summary`, `content`, `child_age`
- **Risk assessment**: `risk_types`, `outcome`, `warning_signs_early`, `risk_factors`, `barriers`
- **Metadata**: `review_date`, `agencies`, `relationship_model`
- **Technical**: `embedding` (vector), `source_file`, `file_hash`, `created_at`, `updated_at`

### `timeline_events` Table
- **Core fields**: `id`, `case_review_id` (foreign key), `event_date`, `event_type`
- **Content**: `description`, `impact`
- **Metadata**: `created_at`, `updated_at`

### `users` Table (Optional)
- **Authentication**: `username`, `email`, `password_hash`, `role`
- **Metadata**: `created_at`, `updated_at`

## Performance Features

- **Indexes**: Created on frequently queried columns
- **JSONB Indexes**: GIN indexes on JSONB columns for efficient searching
- **Vector Index**: Optimized for pgvector similarity searches
- **Foreign Keys**: Proper referential integrity with CASCADE deletes

## Schema Verification

After setup, you can verify the schema by running:

```bash
python setup_database_schema.py
```

Choose option **2** to verify existing schema.

## Troubleshooting

### Common Issues

1. **Connection Error**: Check your `DATABASE_URL` and ensure the database is accessible
2. **Permission Error**: Ensure your database user has CREATE TABLE permissions
3. **pgvector Extension**: The script will automatically create the pgvector extension

### Reset Schema

If you need to start over, you can drop all tables:

```bash
python setup_database_schema.py
```

Choose option **3** (⚠️ **WARNING**: This deletes all data!)

## Next Steps

After successful schema setup:

1. **Run the PDF processor**: `python db_setup_postgresql.py`
2. **Process your first PDF**: The script will now work without column errors
3. **Monitor performance**: Check database performance with the created indexes

## Schema Compatibility

This schema is designed to work with:
- **PostgreSQL 12+** (for JSONB support)
- **pgvector extension** (for embedding storage)
- **The PDF processor script** (`db_setup_postgresql.py`)

## Support

If you encounter issues:
1. Check the error messages for specific details
2. Verify your database connection string
3. Ensure your PostgreSQL user has sufficient privileges
4. Check that pgvector extension can be installed
