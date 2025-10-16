#!/usr/bin/env python3
"""
🌐 APMDM Simplified Web Server

Using refactored single-file multi-class architecture
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Import APMDM generator and visual decoder
from sudoku_generator import APMDMDataManager
from sudoku_verifier import APMDMValidator
from sudoku_visual import TokenVisualDecoder


class APMDMSimpleHandler(BaseHTTPRequestHandler):
    """Simplified HTTP request handler"""
    
    def __init__(self, *args, **kwargs):
        self.data_manager = APMDMDataManager()
        self.validator = APMDMValidator()
        self.visual_decoder = TokenVisualDecoder()
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self._send_cors_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/' or path == '':
                self._serve_index()
            elif path == '/apmdm':
                self._serve_html('apmdm_animation.html')
            elif path.endswith('.html'):
                self._serve_html(path[1:])
            elif path.endswith('.jsonl'):
                self._serve_jsonl(path[1:])
            elif path.endswith('.pkl') or path.endswith('.pkl.gz'):
                self._serve_pkl(path[1:])
            elif path == '/api/health':
                self._send_json({'ok': True, 'message': 'APMDM service running normally'})
            else:
                self._send_404()
        except Exception as e:
            self._send_error(500, str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                request_data = json.loads(body.decode('utf-8'))
            else:
                request_data = {}
            
            if path == '/api/generate_apmdm':
                self._handle_generate_apmdm(request_data)
            elif path == '/api/validate_samples':
                self._handle_validate_samples(request_data)
            elif path == '/api/decode_visual':
                self._handle_decode_visual(request_data)
            else:
                self._send_404()
                
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON format")
        except Exception as e:
            self._send_error(500, str(e))
    
    def _handle_generate_apmdm(self, request_data):
        """Handle APMDM data generation"""
        data_file = request_data.get('data')
        index = request_data.get('index', 0)
        
        if not data_file:
            self._send_json({'ok': False, 'error': 'Missing data parameter'})
            return
        
        output_file = "sudoku_apmdm_with_visual.jsonl"
        # 🔥 Fix: Force JSONL format for frontend visualization compatibility
        result = self.data_manager.generate_from_file_with_format(data_file, index, output_file, format='jsonl')
        
        if result['success']:
            self._send_json({
                'ok': True,
                'sample_count': result['sample_count'],
                'output_file': output_file,
                'message': result.get('message', 'APMDM data generated successfully')
            })
        else:
            self._send_json({
                'ok': False,
                'error': result.get('error', 'Unknown error'),
                'message': result.get('message', 'Generation failed')
            })
    
    def _handle_validate_samples(self, request_data):
        """Handle sample validation"""
        jsonl_file = request_data.get('file', 'sudoku_apmdm_with_visual.jsonl')
        
        if not Path(jsonl_file).exists():
            self._send_json({
                'ok': False,
                'error': f'File not found: {jsonl_file}'
            })
            return
        
        # Load samples
        samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        # Validate samples
        validation_result = self.validator.validate_samples(samples)
        
        self._send_json({
            'ok': True,
            'validation_result': validation_result,
            'message': f'Validation complete, success rate: {validation_result["success_rate"]:.1%}'
        })
    
    def _handle_decode_visual(self, request_data):
        """🔥 Update: Handle token visual decoding, support new format (integer array)"""
        # Support two input formats: token_sequence (string) or sample_data (sample object)
        token_sequence = request_data.get('token_sequence')
        sample_data = request_data.get('sample_data')
        
        if not token_sequence and not sample_data:
            self._send_json({
                'ok': False,
                'error': 'Missing token_sequence or sample_data parameter'
            })
            return
        
        try:
            if sample_data:
                # New format: sample data (supports integer array)
                visual_info = self.visual_decoder.decode_sample_data(sample_data)
            else:
                # Old format: token sequence string
                visual_info = self.visual_decoder.decode_token_sequence(token_sequence)
            
            self._send_json({
                'ok': True,
                'visual_info': visual_info,
                'message': 'Visual info decoded successfully'
            })
            
        except Exception as e:
            self._send_json({
                'ok': False,
                'error': f'Decoding failed: {str(e)}',
                'message': 'Visual info decoding failed'
            })
    
    def _serve_index(self):
        """Serve homepage - direct redirect to APMDM visualization"""
        self.send_response(302)  # Temporary redirect
        self.send_header('Location', '/apmdm')
        self._send_cors_headers()
        self.end_headers()
    
    def _serve_html(self, filename):
        """Serve HTML file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
            
        except FileNotFoundError:
            self._send_404()
    
    def _serve_jsonl(self, filename):
        """Serve JSONL file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
            
        except FileNotFoundError:
            self._send_404()
    
    def _serve_pkl(self, filename):
        """Serve PKL file - 🚀 Auto-convert to JSON format"""
        try:
            # Load PKL data using APMDMDataManager
            samples = self.data_manager.load_samples(filename)
            
            # Convert to frontend-compatible format (list format)
            json_samples = []
            for sample in samples:
                # Convert numpy arrays to lists
                json_sample = self.visual_decoder.sample_to_array_format(sample)
                json_samples.append(json_sample)
            
            # Send JSON response
            self._send_json(json_samples)
            
        except FileNotFoundError:
            self._send_404()
        except Exception as e:
            print(f"PKL service error: {e}")
            self._send_json({'error': f'PKL file processing failed: {str(e)}'})
    
    def _send_json(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._send_cors_headers()
        self.end_headers()
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(json_str.encode('utf-8'))
    
    def _send_html(self, html):
        """Send HTML response"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def _send_error(self, code, message):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._send_cors_headers()
        self.end_headers()
        
        error_response = {'ok': False, 'error': message, 'code': code}
        json_str = json.dumps(error_response, ensure_ascii=False, indent=2)
        self.wfile.write(json_str.encode('utf-8'))
    
    def _send_404(self):
        """Send 404 response"""
        self._send_error(404, "Page not found")
    
    def _send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Custom log format"""
        pass  # Use silent logging


def run_server(port: int = 8001):
    """Start APMDM server"""
    
    print(f"🚀 Starting APMDM refactored server")
    print(f"📡 Port: {port}")
    print(f"🌐 Access URLs:")
    print(f"   Homepage: http://localhost:{port}/")
    print(f"   APMDM Visualization: http://localhost:{port}/apmdm")
    print(f"   Health Check: http://localhost:{port}/api/health")
    print(f"\n📋 Refactored Architecture:")
    print(f"   ✅ APMDMSampleGenerator - Event listening and sample generation")
    print(f"   ✅ APMDMDataManager - High-level data management")
    print(f"   ✅ APMDMValidator - Sample validation")
    print()
    
    try:
        server = HTTPServer(('', port), APMDMSimpleHandler)
        print(f"✅ Server started successfully, press Ctrl+C to stop")
        server.serve_forever()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='APMDM Refactored Web Server')
    parser.add_argument('--port', type=int, default=8001, help='Server port (default: 8001)')
    
    args = parser.parse_args()

    print("🎯 APMDM Refactored - Single-file multi-class architecture")
    print("=" * 40)
    
    # Check required files
    required_files = ['apmdm_animation.html', 'sudoku_loader.py', 'sudoku_solver.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"⚠️ Warning: Missing files: {missing_files}")
        print("   Server can still start, but some features may be limited")
        print()
    
    run_server(args.port)


if __name__ == '__main__':
    main()
