// API Connectivity Test Utility
// Run this script using Node.js to test connectivity to the Flask API

async function testApiConnectivity() {
  const API_BASE_URL = 'http://localhost:5000';
  
  console.log('Testing API connectivity to:', API_BASE_URL);
  
  try {
    // Test health endpoint
    console.log('\nTesting health endpoint...');
    const healthResponse = await fetch(`${API_BASE_URL}/health`);
    const healthStatus = healthResponse.status;
    console.log('Health endpoint status:', healthStatus);
    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      console.log('Health endpoint data:', healthData);
    } else {
      console.error('Health endpoint error:', healthStatus);
    }
    
    // Test available letters endpoint
    console.log('\nTesting letters endpoint...');
    const lettersResponse = await fetch(`${API_BASE_URL}/letters`);
    const lettersStatus = lettersResponse.status;
    console.log('Letters endpoint status:', lettersStatus);
    if (lettersResponse.ok) {
      const lettersData = await lettersResponse.json();
      console.log('Letters endpoint data:', lettersData);
    } else {
      console.error('Letters endpoint error:', lettersStatus);
    }
    
    console.log('\nAPI connectivity test completed.');
  } catch (error) {
    console.error('API connectivity test failed:', error);
  }
}

// Run the test
testApiConnectivity();