#!/usr/bin/env python3
"""
SignBridge Platform Setup Script
Automates the setup and deployment of the SignBridge platform with dual Firebase accounts.
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, Any

class SignBridgeSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.frontend_dir = self.root_dir / "sign-talk-pal"
        self.backend_dir = self.root_dir / "BackEnd"
        
    def run_command(self, command: str, cwd: Path = None, check: bool = True):
        """Run shell command and return result"""
        print(f"Running: {command}")
        if cwd:
            print(f"In directory: {cwd}")
        
        result = subprocess.run(
            command.split(),
            cwd=cwd or self.root_dir,
            capture_output=True,
            text=True,
            check=check
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result
    
    def check_prerequisites(self):
        """Check if all required tools are installed"""
        print("🔍 Checking prerequisites...")
        
        prerequisites = {
            "node": "Node.js",
            "npm": "npm",
            "python": "Python",
            "pip": "pip",
            "firebase": "Firebase CLI"
        }
        
        missing = []
        for cmd, name in prerequisites.items():
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, check=True)
                print(f"✅ {name}: OK")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(name)
                print(f"❌ {name}: Missing")
        
        if missing:
            print(f"\n❌ Missing prerequisites: {', '.join(missing)}")
            print("Please install missing tools and try again.")
            sys.exit(1)
        
        print("✅ All prerequisites installed!")
    
    def setup_frontend(self, skip_deps: bool = False):
        """Setup frontend application"""
        print("\n🎨 Setting up frontend...")
        
        if not self.frontend_dir.exists():
            print(f"❌ Frontend directory not found: {self.frontend_dir}")
            sys.exit(1)
        
        # Install dependencies
        if not skip_deps:
            print("📦 Installing frontend dependencies...")
            self.run_command("npm install", cwd=self.frontend_dir)
        
        # Copy environment file if it doesn't exist
        env_example = self.frontend_dir / ".env.example"
        env_local = self.frontend_dir / ".env.local"
        
        if env_example.exists() and not env_local.exists():
            print("📄 Creating .env.local from .env.example...")
            env_local.write_text(env_example.read_text())
            print("⚠️  Please update .env.local with your configuration")
        
        print("✅ Frontend setup complete!")
    
    def setup_backend(self, skip_deps: bool = False):
        """Setup backend services"""
        print("\n⚙️ Setting up backend...")
        
        if not self.backend_dir.exists():
            print(f"❌ Backend directory not found: {self.backend_dir}")
            sys.exit(1)
        
        # Install Python dependencies
        if not skip_deps:
            print("📦 Installing backend dependencies...")
            requirements_file = self.backend_dir / "functions" / "requirements.txt"
            if requirements_file.exists():
                self.run_command(f"pip install -r {requirements_file}")
            else:
                print("⚠️  requirements.txt not found, skipping Python dependencies")
        
        # Copy environment file if it doesn't exist
        env_example = self.backend_dir / ".env.example"
        env_file = self.backend_dir / ".env"
        
        if env_example.exists() and not env_file.exists():
            print("📄 Creating .env from .env.example...")
            env_file.write_text(env_example.read_text())
            print("⚠️  Please update .env with your configuration")
        
        print("✅ Backend setup complete!")
    
    def setup_firebase_hosting(self):
        """Setup Firebase hosting for frontend"""
        print("\n🔥 Setting up Firebase hosting...")
        
        # Check if Firebase is already initialized
        firebase_json = self.frontend_dir / "firebase.json"
        if not firebase_json.exists():
            print("🚀 Initializing Firebase hosting...")
            self.run_command("firebase init hosting", cwd=self.frontend_dir)
        else:
            print("✅ Firebase hosting already initialized")
        
        print("✅ Firebase hosting setup complete!")
    
    def setup_firebase_functions(self):
        """Setup Firebase functions for backend"""
        print("\n🔥 Setting up Firebase functions...")
        
        # Check if Firebase is already initialized
        firebase_json = self.backend_dir / "firebase.json"
        if not firebase_json.exists():
            print("🚀 Initializing Firebase functions...")
            self.run_command("firebase init functions", cwd=self.backend_dir)
        else:
            print("✅ Firebase functions already initialized")
        
        print("✅ Firebase functions setup complete!")
    
    def build_frontend(self):
        """Build frontend for production"""
        print("\n🏗️ Building frontend...")
        self.run_command("npm run build", cwd=self.frontend_dir)
        print("✅ Frontend build complete!")
    
    def deploy_frontend(self):
        """Deploy frontend to Firebase hosting"""
        print("\n🚀 Deploying frontend...")
        self.run_command("firebase deploy --only hosting", cwd=self.frontend_dir)
        print("✅ Frontend deployed!")
    
    def deploy_backend(self):
        """Deploy backend to Firebase functions"""
        print("\n🚀 Deploying backend...")
        self.run_command("firebase deploy --only functions", cwd=self.backend_dir)
        print("✅ Backend deployed!")
    
    def run_dev_servers(self):
        """Run development servers"""
        print("\n🔧 Starting development servers...")
        
        # Start backend in background
        print("Starting backend server...")
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "functions.main:main_app", 
            "--reload", "--host", "0.0.0.0", "--port", "8000"
        ], cwd=self.backend_dir)
        
        # Start frontend
        print("Starting frontend server...")
        try:
            self.run_command("npm run dev", cwd=self.frontend_dir)
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            backend_process.terminate()
            backend_process.wait()
    
    def run_tests(self):
        """Run all tests"""
        print("\n🧪 Running tests...")
        
        # Frontend tests
        print("Running frontend tests...")
        try:
            self.run_command("npm test", cwd=self.frontend_dir, check=False)
        except subprocess.CalledProcessError:
            print("⚠️  Frontend tests failed or not configured")
        
        # Backend tests
        print("Running backend tests...")
        try:
            self.run_command("python -m pytest", cwd=self.backend_dir, check=False)
        except subprocess.CalledProcessError:
            print("⚠️  Backend tests failed or not configured")
        
        print("✅ Tests complete!")

def main():
    parser = argparse.ArgumentParser(description="SignBridge Platform Setup")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-firebase", action="store_true", help="Skip Firebase initialization")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Full setup")
    
    # Dev command
    dev_parser = subparsers.add_parser("dev", help="Start development servers")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build for production")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to production")
    deploy_parser.add_argument("--frontend-only", action="store_true", help="Deploy frontend only")
    deploy_parser.add_argument("--backend-only", action="store_true", help="Deploy backend only")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup = SignBridgeSetup()
    
    if args.command == "setup":
        setup.check_prerequisites()
        setup.setup_frontend(skip_deps=args.skip_deps)
        setup.setup_backend(skip_deps=args.skip_deps)
        if not args.skip_firebase:
            setup.setup_firebase_hosting()
            setup.setup_firebase_functions()
        print("\n🎉 Setup complete! Run 'python setup.py dev' to start development servers.")
    
    elif args.command == "dev":
        setup.run_dev_servers()
    
    elif args.command == "build":
        setup.build_frontend()
        print("\n🎉 Build complete! Ready for deployment.")
    
    elif args.command == "deploy":
        if args.frontend_only:
            setup.build_frontend()
            setup.deploy_frontend()
        elif args.backend_only:
            setup.deploy_backend()
        else:
            setup.build_frontend()
            setup.deploy_frontend()
            setup.deploy_backend()
        print("\n🎉 Deployment complete!")
    
    elif args.command == "test":
        setup.run_tests()

if __name__ == "__main__":
    main()