# YouTube OAuth Setup Guide

This document explains how to set up OAuth 2.0 credentials for YouTube API access in your VideoAI project.

## Error: redirect_uri_mismatch

If you're seeing the error:
```
Error 400: redirect_uri_mismatch
```

This means that the redirect URI in your Google Cloud Console project doesn't match the one used in your application.

## Solution

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Select or create a project
3. Navigate to "APIs & Services" > "Credentials"
4. Find your OAuth 2.0 Client ID or create a new one:
   - Click "Create Credentials" > "OAuth client ID"
   - Select "Web application" as the application type
   - Add a name for your client ID (e.g., "VideoAI YouTube Uploader")
   - Under "Authorized redirect URIs", add: `http://localhost:8080/`
   - Click "Create"

5. Download the client secret JSON file
6. Save it as `client_secret.json` in your project directory

## Important Notes

- The redirect URI must exactly match what's in the code: `http://localhost:8080/`
- Port 8080 is specifically configured in the code
- Make sure you've enabled the YouTube Data API v3 for your project
- If you've already created OAuth credentials, you can edit them to add the correct redirect URI
- Delete any existing token files in your project directory if you've previously authenticated with incorrect settings

## Testing Authentication

Run the upload script directly to test authentication:

```bash
python upload_video.py --channel 1
```

This will open your browser for authentication. After successful authentication, the token will be saved for future use.
