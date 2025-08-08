import { useState, useEffect } from 'react';

interface MobileInfo {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isTouchDevice: boolean;
  screenWidth: number;
  screenHeight: number;
  orientation: 'portrait' | 'landscape';
  isIOS: boolean;
  isAndroid: boolean;
  isStandalone: boolean;
}

export function useMobile(): MobileInfo {
  const [mobileInfo, setMobileInfo] = useState<MobileInfo>(() => {
    if (typeof window === 'undefined') {
      return {
        isMobile: false,
        isTablet: false,
        isDesktop: true,
        isTouchDevice: false,
        screenWidth: 1024,
        screenHeight: 768,
        orientation: 'landscape',
        isIOS: false,
        isAndroid: false,
        isStandalone: false,
      };
    }

    return getMobileInfo();
  });

  useEffect(() => {
    const handleResize = () => {
      setMobileInfo(getMobileInfo());
    };

    const handleOrientationChange = () => {
      // Delay to get accurate dimensions after orientation change
      setTimeout(() => {
        setMobileInfo(getMobileInfo());
      }, 100);
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleOrientationChange);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleOrientationChange);
    };
  }, []);

  return mobileInfo;
}

function getMobileInfo(): MobileInfo {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const userAgent = navigator.userAgent;
  
  // Device detection
  const isMobileUserAgent = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent);
  const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  
  // Screen size based detection
  const isMobileScreen = width < 768;
  const isTabletScreen = width >= 768 && width < 1024;
  
  // Combine user agent and screen size for more accurate mobile detection
  const isMobile = isMobileScreen || (isMobileUserAgent && width < 1024);
  const isTablet = (isTabletScreen && isTouchDevice) || (isMobileUserAgent && width >= 768 && width < 1024);
  const isDesktop = !isMobile && !isTablet;
  
  // Platform detection
  const isIOS = /iPad|iPhone|iPod/.test(userAgent);
  const isAndroid = /Android/.test(userAgent);
  
  // PWA detection
  const isStandalone = window.matchMedia('(display-mode: standalone)').matches || 
                      (window.navigator as any).standalone === true;
  
  // Orientation
  const orientation = height > width ? 'portrait' : 'landscape';
  
  return {
    isMobile,
    isTablet,
    isDesktop,
    isTouchDevice,
    screenWidth: width,
    screenHeight: height,
    orientation,
    isIOS,
    isAndroid,
    isStandalone,
  };
}