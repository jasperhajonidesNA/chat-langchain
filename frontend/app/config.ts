"use client";

// This file centralizes all environment configuration with debugging
// We use 'use client' to ensure it works in client components

// Check if we're in browser environment
const isBrowser = typeof window !== 'undefined';

// RUNTIME PATCHING - Override any incorrect URLs
if (isBrowser) {
  // This will run in the browser and patch any hard-coded URL references
  const originalFetch = window.fetch;
  window.fetch = function(input, init) {
    // Check if this is a localhost:2024 URL
    if (typeof input === 'string' && input.includes('localhost:2024')) {
      // Replace with the correct API URL
      const correctApiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL || 'https://naturealpha-docs-6f63c1e32335558f86984cf800d1f815.us.langgraph.app';
      
      // Replace the URL
      const newUrl = input.replace(/http:\/\/localhost:2024/g, correctApiUrl);
      
      // Call the original fetch with the corrected URL
      return originalFetch(newUrl, init);
    }
    
    // Otherwise, proceed normally
    return originalFetch(input, init);
  };
}

export const ENV = {
  // Public API URL (accessible in client components)
  API_URL: (() => {
    // Prioritize NEXT_PUBLIC_API_URL, then API_BASE_URL, no localhost fallback
    const url = process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL || 'https://naturealpha-docs-6f63c1e32335558f86984cf800d1f815.us.langgraph.app';
    return url;
  })(),

  // API Base URL (accessible in client components via NEXT_PUBLIC prefix)
  API_BASE_URL: (() => {
    const url = process.env.API_BASE_URL || 'https://naturealpha-docs-6f63c1e32335558f86984cf800d1f815.us.langgraph.app';
    return url;
  })(),

  // LangChain API Key
  LANGCHAIN_API_KEY: (() => {
    const key = process.env.LANGCHAIN_API_KEY || '';
    return key;
  })(),

  // Debug function to print all environment variables
  debug: () => {
    console.log('[CONFIG] Environment Variables:');
    console.log('  NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);
    console.log('  API_BASE_URL:', process.env.API_BASE_URL);
    console.log('  LANGCHAIN_API_KEY available:', !!process.env.LANGCHAIN_API_KEY);
    console.log('  NODE_ENV:', process.env.NODE_ENV);
    
    if (isBrowser) {
      console.log('  Running in browser at:', window.location.href);
    }
  }
};

// Print debug info on load
ENV.debug(); 