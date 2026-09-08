import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import { ConfigInterface } from '../interfaces/Config';
import { toast } from "sonner";

interface ConfigProviderProps {
  children: ReactNode;
}

interface ConfigDataContextType {
  config: ConfigInterface | null;
  refreshConfigData: () => void;
}

const ConfigContext = createContext<ConfigDataContextType | null>(null);

export const ConfigProvider: React.FC<ConfigProviderProps> = ({ children }) => {
  const [config, setConfig] = useState<ConfigInterface | null>(null);
  const [refreshData, setRefreshData] = useState<boolean>(false);

  useEffect(() => {
    fetchConfigData().then((data) => {
      setConfig(data);
    });
  }, [refreshData]);

  const fetchConfigData = async () => {
    try {
      const response = await fetch('/api/config');
      if (!response.ok) {
        throw new Error('');
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(error);
      toast.error(`Error while fetching config data: ${error}`);
      return null;
    }
  };

  const refreshConfigData = () => {
    setRefreshData(!refreshData);
  };


  return (
    <ConfigContext.Provider value={{config, refreshConfigData}}>
      {children}
    </ConfigContext.Provider>
  );
};

export const useConfigData = () => {
  return useContext(ConfigContext);
};

export const updateConfigData = async (configData: ConfigInterface): Promise<boolean> => {
  try {
    const response = await fetch('/api/config', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(configData),
    });
    const response_text = await response.text();
    if (!response.ok) {
      // The body carries the validation message (e.g. the 400 raised when a
      // joiner participant id fails the id pattern) - surface it verbatim
      // instead of a generic error.
      toast.error(response_text);
      return false;
    }
    toast.info(response_text);
    return true;
  } catch (error) {
    console.error(error);
    toast.error(`Error while sending config data to backend: ${error}`);
    return false;
  }
};
