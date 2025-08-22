import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Header } from '@/components/header';
import { useAuthStore } from '@/lib/auth';
import { usePreferencesStore } from '@/lib/preferences';
import { ArrowLeft, User, Settings, Shield, History } from 'lucide-react';

export default function ProfilePage() {
  const { user, logout } = useAuthStore();
  const [, setLocation] = useLocation();
  const { searchHistoryEnabled, setSearchHistoryEnabled } = usePreferencesStore();

  const handleLogout = () => {
    logout();
    setLocation('/login');
  };

  return (
    <div className="min-h-screen bg-neutral-50">
      <Header />
      
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6">
          <Button
            variant="ghost"
            onClick={() => setLocation('/')}
            className="mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Search
          </Button>
          
          <h1 className="text-3xl font-bold text-gray-900">Profile Settings</h1>
          <p className="text-gray-600 mt-2">Manage your account preferences and settings</p>
        </div>

        <div className="space-y-6">
          {/* User Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <User className="w-5 h-5 mr-2" />
                User Information
              </CardTitle>
              <CardDescription>
                Your account details and role information
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-sm font-medium text-gray-700">Name</Label>
                  <p className="text-gray-900 mt-1">{user?.name}</p>
                </div>
                <div>
                  <Label className="text-sm font-medium text-gray-700">Username</Label>
                  <p className="text-gray-900 mt-1">{user?.username}</p>
                </div>
                <div>
                  <Label className="text-sm font-medium text-gray-700">Role</Label>
                  <p className="text-gray-900 mt-1">{user?.role}</p>
                </div>
                <div>
                  <Label className="text-sm font-medium text-gray-700">User ID</Label>
                  <p className="text-gray-900 mt-1">{user?.id}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Privacy Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Shield className="w-5 h-5 mr-2" />
                Privacy & Data Settings
              </CardTitle>
              <CardDescription>
                Control how your data is handled and stored
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label className="text-sm font-medium text-gray-900">Search History</Label>
                  <p className="text-sm text-gray-500">
                    Enable to store your search queries for quick access to recent searches
                  </p>
                </div>
                <Switch
                  checked={searchHistoryEnabled}
                  onCheckedChange={setSearchHistoryEnabled}
                />
              </div>
              
              {searchHistoryEnabled && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <History className="w-5 h-5 text-blue-600 mr-3 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-blue-800">
                      <p className="font-medium">Search History Enabled</p>
                      <p className="mt-1">
                        Your search queries will be stored locally and can be accessed through the Search History button. 
                        This data is stored only on your device and is not shared with the server.
                      </p>
                    </div>
                  </div>
                </div>
              )}
              
              {!searchHistoryEnabled && (
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <Shield className="w-5 h-5 text-gray-600 mr-3 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-gray-700">
                      <p className="font-medium">Search History Disabled</p>
                      <p className="mt-1">
                        Search queries are not stored for privacy protection. The Search History button will not be visible.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Account Actions</CardTitle>
              <CardDescription>
                Manage your account and session
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button
                variant="destructive"
                onClick={handleLogout}
                className="w-full sm:w-auto"
              >
                Sign Out
              </Button>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
