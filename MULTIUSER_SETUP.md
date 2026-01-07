# Multi-User Support - Setup Guide

This guide explains how to set up and use the new multi-user functionality in AI Career Coach.

## What's New?

AI Career Coach now supports multiple users with isolated data:
- ✅ User authentication (registration, login, logout)
- ✅ User-specific resume storage
- ✅ User-specific FAISS vector indices
- ✅ User-specific job matches
- ✅ User profile/dashboard
- ✅ Secure password hashing
- ✅ Session-based authentication

## Architecture Changes

### New Database Tables
- `users` - User accounts with authentication
- `resumes` - Resume metadata linked to users
- `job_matches` - Now includes `user_id` and `resume_id` foreign keys

### File Structure Changes
```
uploads/
├── 1/               # User ID 1's resumes
│   └── resume.pdf
├── 2/               # User ID 2's resumes
│   └── resume.pdf

vector_index/
├── 1/               # User ID 1's FAISS indices
│   ├── index.faiss
│   └── index.pkl
├── 2/               # User ID 2's FAISS indices
    ├── index.faiss
    └── index.pkl
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `Flask-Login` - Session management
- `Flask-WTF` - Form handling and CSRF protection
- `email-validator` - Email validation

### 2. Run Database Migration

**IMPORTANT:** Backup your existing `career_coach.db` before migration!

```bash
# Backup existing database (optional but recommended)
cp career_coach.db career_coach.db.backup

# Run migration script (standalone version - no heavy dependencies needed)
python migrate_standalone.py
```

**Note:** Use `migrate_standalone.py` instead of `migrate_to_multiuser.py` to avoid installing langchain dependencies just for migration.

The migration script will:
1. Create new tables (`users`, `resumes`)
2. Add new columns to `job_matches`
3. Create a default admin user
4. Migrate existing data to the new schema
5. Move files to user-specific directories

### 3. Default Admin Account

After migration, you can login with:
- **Email:** `admin@aicareercoach.local`
- **Password:** `admin123`

**⚠️ IMPORTANT:** Change this password immediately after first login!

## Usage

### For End Users

#### 1. Registration
- Visit `/register`
- Fill in username, email, and password
- Click "Register"

#### 2. Login
- Visit `/login`
- Enter email and password
- Check "Remember Me" for persistent sessions

#### 3. Upload Resume
- After login, you're redirected to the home page
- Upload your resume (PDF format)
- View analysis and job matches

#### 4. Profile Dashboard
- Click "Profile" in navigation
- View all your uploaded resumes
- See recent job matches
- Check your statistics

#### 5. Logout
- Click "Logout" in navigation

### Data Isolation

Each user has:
- **Separate resume storage:** `uploads/{user_id}/`
- **Separate vector indices:** `vector_index/{user_id}/`
- **Isolated job matches:** Only sees their own matches
- **Private Q&A:** Questions answered from their own resumes

### Security Features

1. **Password Security**
   - Passwords are hashed using `werkzeug.security`
   - Never stored in plain text

2. **Session Management**
   - Flask-Login handles user sessions
   - Automatic session expiry
   - "Remember Me" functionality

3. **Access Control**
   - All routes protected with `@login_required`
   - Users can only access their own data
   - CSRF protection on forms

## API Changes

### Updated Endpoints

#### `/api/matches/<resume_id>` (Changed)
- Now requires `resume_id` instead of `filename`
- Validates that resume belongs to current user

#### All routes now require authentication
- Returns redirect to login if not authenticated
- Sets `next` parameter for redirect after login

## Development Notes

### Adding New Features

When adding new user-specific features:

1. **Database Models**
   - Add `user_id` foreign key
   - Add relationship to User model

2. **Routes**
   - Add `@login_required` decorator
   - Use `current_user` for user context
   - Filter queries by `user_id`

3. **File Storage**
   - Use `os.path.join('folder', str(current_user.id))` for user-specific paths
   - Create directories with `os.makedirs(..., exist_ok=True)`

### Example: User-Specific Query

```python
# OLD (single-user)
jobs = JobMatch.query.all()

# NEW (multi-user)
jobs = JobMatch.query.filter_by(user_id=current_user.id).all()

# Or using relationships
jobs = current_user.job_matches.all()
```

## Troubleshooting

### "module 'hashlib' has no attribute 'scrypt'"
- **Fixed!** Use `migrate_standalone.py` which uses `pbkdf2:sha256` instead
- This error occurred with older Python versions or limited OpenSSL builds
- Our code now uses a more compatible hashing algorithm

### "Please log in to access this page"
- You're not authenticated
- Go to `/login` to sign in

### Can't see my old resumes
- Run the migration script
- It will create an admin user and migrate old data

### Migration fails
- Check that `career_coach.db` exists
- Ensure you have write permissions
- Check for Python errors in terminal
- Use `migrate_standalone.py` (lighter dependencies)

### Sessions not persisting
- Set `SECRET_KEY` environment variable for production
- Default key is for development only

## Production Deployment

### Environment Variables

```bash
export SECRET_KEY="your-secret-key-here"  # Generate secure random key
export XAI_API_KEY="your-xai-key"
export ADZUNA_APP_ID="your-adzuna-id"
export ADZUNA_APP_KEY="your-adzuna-key"
```

### Security Checklist

- [ ] Change default admin password
- [ ] Set strong SECRET_KEY
- [ ] Use HTTPS in production
- [ ] Enable secure cookies (SESSION_COOKIE_SECURE=True)
- [ ] Set up proper user registration validation
- [ ] Consider adding email verification
- [ ] Implement rate limiting
- [ ] Add password reset functionality

## Future Enhancements

Potential improvements:
- Email verification for new accounts
- Password reset functionality
- Social login (Google, GitHub, etc.)
- User profile editing
- Account deletion
- Admin panel for user management
- Resume sharing between users
- Team/organization support

## Support

For issues or questions:
- Check the main README.md
- Review this setup guide
- Check the migration script output
- Review Flask-Login documentation: https://flask-login.readthedocs.io/

---

**Version:** 2.0.0 (Multi-User Support)
**Last Updated:** January 2026
