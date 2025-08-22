import { Button } from '@/components/ui/button';
import { useAuthStore } from '@/lib/auth';
import { useLocation } from 'wouter';
import { User, LogOut } from 'lucide-react';

export function Header() {
  const { user, logout } = useAuthStore();
  const [, setLocation] = useLocation();

  const handleProfileClick = () => {
    setLocation('/profile');
  };

  const handleLogout = () => {
    logout();
    setLocation('/login');
  };

  return (
    <header className="bg-white border-b border-neutral-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0">
              <h1 className="text-xl font-semibold text-primary" data-testid="text-header-title">
                NSPCC Case Review Search
              </h1>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="text-sm text-neutral-600" data-testid="text-user-info">
              Logged in as: {user?.name}, Social Worker
            </span>
            
            <Button 
              variant="ghost" 
              size="sm"
              className="text-neutral-700 hover:bg-neutral-100"
              data-testid="button-profile"
              onClick={handleProfileClick}
            >
              <User className="w-4 h-4 mr-2" />
              Profile
            </Button>
            
            <Button
              variant="destructive"
              size="sm"
              onClick={handleLogout}
              data-testid="button-logout"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
}
