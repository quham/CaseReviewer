import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface PreferencesStore {
  searchHistoryEnabled: boolean;
  setSearchHistoryEnabled: (enabled: boolean) => void;
}

export const usePreferencesStore = create<PreferencesStore>()(
  persist(
    (set) => ({
      searchHistoryEnabled: false,
      setSearchHistoryEnabled: (enabled: boolean) => set({ searchHistoryEnabled: enabled }),
    }),
    {
      name: 'user-preferences',
    }
  )
);
