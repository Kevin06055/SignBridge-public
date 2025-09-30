# Text-to-Sign Integration Testing Guide

This guide will help you test the integration between the React frontend and Flask backend for the Text-to-Sign functionality.

## Prerequisites

1. Make sure you have both applications running:

   ### Start the Flask backend:
   ```powershell
   # Navigate to the text-to-sign directory
   cd d:\MyWorks\SignBridge\text-to-sign
   
   # Start the Flask server
   python app.py
   ```

   The Flask server should be running on http://localhost:5000

   ### Start the React frontend:
   ```powershell
   # Navigate to the sign-talk-pal directory
   cd d:\MyWorks\SignBridge\sign-talk-pal
   
   # Install dependencies if you haven't already
   npm install
   
   # Start the development server
   npm run dev
   ```

   The React app should be running on http://localhost:5173 (or another port if specified)

## Testing Steps

1. **Basic Connectivity Test**:
   - Open the React app in your browser
   - Navigate to the "Speech to Sign" page
   - Check that the API status indicator shows "Sign Language API Connected" with green status
   - If it shows "Sign Language API Disconnected" in red, ensure the Flask server is running

2. **Text-to-Sign Conversion Test**:
   - Enter some text in the input field (e.g., "Hello world")
   - Click the "Play Sign Animation" button
   - Verify that a video is generated and displayed in the video player
   - The video should show sign language representations of the entered text

3. **Error Handling Test**:
   - Try stopping the Flask server while the React app is running
   - Refresh the page or navigate away and back to the Speech to Sign page
   - Verify that the API status indicator shows "Sign Language API Disconnected" in red
   - Try to convert text by clicking the "Play Sign Animation" button
   - Verify that appropriate error messages are displayed

4. **Speech Recognition Test**:
   - Click the "Start Speaking" button
   - Speak clearly into your microphone
   - Verify that your speech is transcribed into the text input field
   - Click the "Play Sign Animation" button
   - Verify that a video is generated based on the transcribed text

## Troubleshooting

- **CORS Issues**: If you encounter CORS errors, you may need to modify the Flask backend to allow cross-origin requests.
- **Missing Video Files**: If videos don't appear, check the console for errors and ensure the Flask server is generating videos correctly.
- **API Connection Issues**: If the React app can't connect to the Flask API, ensure the API is running on port 5000 and the API_BASE_URL in the textToSignService.ts file is set correctly.

## Next Steps

Once the integration is working correctly, you might want to:

1. Add more features like saving favorite signs
2. Improve error handling and user feedback
3. Add support for more complex sign language expressions
4. Deploy both applications to a production environment