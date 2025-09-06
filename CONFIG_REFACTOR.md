# 🔧 Configuration Management Refactoring

## 📋 **Problem Solved**
The original `storyboard_to_audio.py` had configuration values scattered throughout the code, making it difficult to manage, test, and customize. Hard-coded defaults were mixed with environment variable handling, creating maintenance challenges.

## 🎯 **Solution: Centralized AudioGenerationConfig**

### **Before (Scattered Configuration)**
```python
# Scattered throughout the file
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ELEVEN_API_URL = os.getenv("ELEVEN_API_URL", "https://api.elevenlabs.io/v1/sound-generation")
DEFAULT_MUSIC_FORMAT = "mp3"
DEFAULT_SFX_FORMAT = "wav"

# In functions
max_retries=4, backoff=1.6, timeout=120

# In argparse
parser.add_argument("--gemini-temp", type=float, default=0.6)
```

### **After (Centralized Pydantic Configuration)**
```python
class AudioGenerationConfig(BaseModel):
    """Centralized configuration for audio generation pipeline using Pydantic"""
    
    # API Configuration
    gemini_api_key: str = ""
    eleven_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    eleven_api_url: str = "https://api.elevenlabs.io/v1/sound-generation"
    
    # Audio Format Defaults
    default_music_format: str = "mp3"
    default_sfx_format: str = "wav"
    
    # Generation Parameters
    gemini_temperature: float = 0.6
    gemini_max_tokens: int = 4096
    rate_limit_sleep: float = 0.7
    
    # Network & Retry Configuration
    max_retries: int = 4
    retry_backoff: float = 1.6
    request_timeout: int = 120
    
    # Pydantic Field definitions with validation
    gemini_temperature: float = Field(
        0.6, 
        ge=0.0, 
        le=2.0, 
        description="Temperature for Gemini generation (0.0-2.0)"
    )
    
    @validator('default_music_format', 'default_sfx_format')
    def validate_audio_formats(cls, v):
        """Validate audio format is supported"""
        valid_formats = {'mp3', 'wav', 'flac', 'aac', 'ogg'}
        if v.lower() not in valid_formats:
            raise ValueError(f"Audio format must be one of: {', '.join(valid_formats)}")
        return v.lower()
    
    @root_validator
    def validate_required_keys_for_production(cls, values):
        """Validate required API keys"""
        if not values.get('gemini_api_key'):
            raise ValueError("gemini_api_key is required")
        return values
    
    @classmethod
    def from_env(cls) -> 'AudioGenerationConfig':
        """Create configuration from environment variables with fallbacks"""
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            # ... all environment mappings
        )
```

## 🚀 **Key Benefits**

### **1. Pydantic-Powered Configuration**
- ✅ **Automatic Validation**: Built-in type checking and constraint validation
- ✅ **Serialization**: JSON/dict export and import built-in
- ✅ **Schema Generation**: Auto-generated OpenAPI-compatible schemas
- ✅ **Field Documentation**: Rich field descriptions and metadata
- ✅ **Type Safety**: Full type hints with runtime validation

### **2. Single Source of Truth**
- All configuration in one place
- Clear documentation of all options
- Type hints for better IDE support

### **3. Environment Variable Support**
```bash
# All configurable via environment
export GEMINI_TEMPERATURE=0.8
export DEFAULT_MUSIC_FORMAT=wav
export RATE_LIMIT_SLEEP=1.0
```

### **3. Programmatic Customization**
```python
# Easy custom configurations
config = create_custom_config(
    gemini_temperature=0.8,
    default_music_format="wav",
    rate_limit_sleep=1.0
)
```

### **4. Runtime Validation**
```python
# Catches configuration errors early
try:
    CONFIG.validate()
except ValueError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)
```

### **5. Better Testing**
```python
# Easy to create test configurations
test_config = create_custom_config(
    gemini_api_key="test-key",
    eleven_api_key="test-key",
    rate_limit_sleep=0.0  # No delays in tests
)
```

## 📚 **Usage Examples**

### **Basic Usage (Global CONFIG)**
```python
from storyboard_to_audio import CONFIG

print(f"Using model: {CONFIG.gemini_model}")
print(f"Music format: {CONFIG.default_music_format}")
```

### **Custom Configuration**
```python
from storyboard_to_audio import create_custom_config

# High-quality, slower processing
hq_config = create_custom_config(
    default_music_format="wav",
    gemini_temperature=0.3,
    rate_limit_sleep=2.0
)

# Fast processing for development
dev_config = create_custom_config(
    rate_limit_sleep=0.1,
    max_retries=1,
    gemini_temperature=1.0
)
```

### **Environment-Based Configuration**
```bash
# Production settings
export GEMINI_MODEL=gemini-2.5-flash
export DEFAULT_MUSIC_FORMAT=mp3
export RATE_LIMIT_SLEEP=1.0
export MAX_RETRIES=5

python storyboard_to_audio.py --storyboard story.txt
```

## 🔄 **Migration Impact**

### **Backward Compatibility**
- ✅ All command-line arguments work the same
- ✅ All environment variables still supported
- ✅ Default behavior unchanged

### **Code Changes**
- ✅ Replaced scattered constants with `CONFIG.property`
- ✅ Added validation at startup
- ✅ Improved error messages
- ✅ Enhanced documentation

## 🎯 **Configuration Options**

| Category | Setting | Environment Variable | Default | Description |
|----------|---------|---------------------|---------|-------------|
| **API** | `gemini_api_key` | `GEMINI_API_KEY` | `""` | Google Gemini API key (required) |
| **API** | `eleven_api_key` | `ELEVEN_API_KEY` | `""` | ElevenLabs API key (required) |
| **API** | `gemini_model` | `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to use |
| **API** | `eleven_api_url` | `ELEVEN_API_URL` | `https://api.elevenlabs.io/v1/sound-generation` | ElevenLabs API endpoint |
| **Audio** | `default_music_format` | `DEFAULT_MUSIC_FORMAT` | `mp3` | Default music file format |
| **Audio** | `default_sfx_format` | `DEFAULT_SFX_FORMAT` | `wav` | Default SFX file format |
| **Generation** | `gemini_temperature` | `GEMINI_TEMPERATURE` | `0.6` | Temperature for Gemini (0.0-2.0) |
| **Generation** | `gemini_max_tokens` | `GEMINI_MAX_TOKENS` | `4096` | Max tokens for Gemini |
| **Network** | `rate_limit_sleep` | `RATE_LIMIT_SLEEP` | `0.7` | Sleep between API calls |
| **Network** | `max_retries` | `MAX_RETRIES` | `4` | Number of retry attempts |
| **Network** | `retry_backoff` | `RETRY_BACKOFF` | `1.6` | Backoff multiplier for retries |
| **Network** | `request_timeout` | `REQUEST_TIMEOUT` | `120` | Request timeout in seconds |

## ✅ **Validation Rules**
- `gemini_api_key` and `eleven_api_key` are required
- `gemini_temperature` must be between 0.0 and 2.0
- `gemini_max_tokens` must be positive
- `rate_limit_sleep` must be non-negative
- Duration constraints: 1 ≤ min_duration ≤ max_duration ≤ 60

## 🎉 **Result**
The refactored configuration system provides:
- ✅ **Maintainability**: Single place to manage all settings
- ✅ **Flexibility**: Easy customization for different use cases
- ✅ **Reliability**: Runtime validation prevents configuration errors
- ✅ **Documentation**: Clear, comprehensive configuration reference
- ✅ **Testing**: Easy to create test configurations
- ✅ **Type Safety**: Full type hints for better development experience
