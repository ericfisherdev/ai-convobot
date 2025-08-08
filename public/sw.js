const CACHE_NAME = 'ai-companion-v1';
const STATIC_CACHE_URLS = [
  '/',
  '/assets/index-4rust.js',
  '/assets/index-4rust2.js', 
  '/assets/index-4rust.css',
  '/ai_companion_logo.jpg',
  '/assets/companion_avatar-4rust.jpg',
  '/manifest.json'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Caching static assets');
        return cache.addAll(STATIC_CACHE_URLS);
      })
      .then(() => {
        console.log('Service Worker installed');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('Failed to cache static assets:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('Service Worker activated');
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', (event) => {
  const { request } = event;
  
  // Handle API requests with network-first strategy
  if (request.url.includes('/api/')) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Clone response for caching
          const responseClone = response.clone();
          
          // Cache successful API responses
          if (response.status === 200) {
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(request, responseClone);
            });
          }
          
          return response;
        })
        .catch(() => {
          // Fallback to cache for offline support
          return caches.match(request).then((response) => {
            if (response) {
              return response;
            }
            // Return offline message for uncached API requests
            return new Response(
              JSON.stringify({ error: 'Offline - cached response not available' }),
              {
                status: 503,
                statusText: 'Service Unavailable',
                headers: { 'Content-Type': 'application/json' }
              }
            );
          });
        })
    );
    return;
  }
  
  // Handle static assets with cache-first strategy
  event.respondWith(
    caches.match(request).then((response) => {
      if (response) {
        return response;
      }
      
      return fetch(request).then((response) => {
        // Don't cache non-successful responses
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }
        
        // Clone the response for caching
        const responseClone = response.clone();
        
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(request, responseClone);
        });
        
        return response;
      });
    })
  );
});

// Background sync for offline message sending
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync-messages') {
    event.waitUntil(
      // Get pending messages from IndexedDB and send them
      sendPendingMessages()
    );
  }
});

async function sendPendingMessages() {
  try {
    // This would integrate with your message sending logic
    console.log('Attempting to send pending messages...');
    
    // In a real implementation, you'd:
    // 1. Get pending messages from IndexedDB
    // 2. Try to send each one
    // 3. Remove successful sends from the pending queue
    // 4. Keep failed sends for next sync attempt
    
  } catch (error) {
    console.error('Failed to send pending messages:', error);
  }
}

// Push notifications (for future use)
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'New message from AI Companion',
    icon: '/ai_companion_logo.jpg',
    badge: '/ai_companion_logo.jpg',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'Open Chat',
        icon: '/ai_companion_logo.jpg'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/ai_companion_logo.jpg'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('AI Companion', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'explore') {
    // Open the app
    event.waitUntil(
      clients.matchAll().then((clientList) => {
        if (clientList.length > 0) {
          return clientList[0].focus();
        }
        return clients.openWindow('/');
      })
    );
  }
});