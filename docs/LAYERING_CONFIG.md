# Layering Configuration Guide

## Overview

The layering order determines which garments are fitted first (innermost) and which are fitted last (outermost). This is configurable via a JSON file.

## Default Layering Order

The system comes with sensible defaults:

```json
{
  "underwear": 0,
  "undershirt": 1,
  "socks": 2,
  "shirt": 3,
  "t-shirt": 4,
  "dress": 5,
  "pants": 6,
  "shorts": 7,
  "skirt": 8,
  "jacket": 9,
  "coat": 10,
  "sweater": 11,
  "shoes": 12,
  "boots": 13,
  "hat": 14,
  "belt": 15
}
```

**Lower numbers = closer to body (fitted first)**
**Higher numbers = further from body (fitted last)**

## Custom Configuration

### Creating a Custom Config File

1. Create a JSON file (e.g., `layering_config.json`):

```json
{
  "underwear": 0,
  "undershirt": 1,
  "socks": 2,
  "shirt": 3,
  "t-shirt": 4,
  "dress": 5,
  "pants": 6,
  "shorts": 7,
  "skirt": 8,
  "jacket": 9,
  "coat": 10,
  "sweater": 11,
  "shoes": 12,
  "boots": 13,
  "hat": 14,
  "belt": 15,
  "vest": 2.5,
  "tie": 3.5
}
```

2. Set environment variable:

```bash
export FITTING_ROOM_LAYERING_CONFIG_PATH=/path/to/layering_config.json
```

Or in `.env` file:

```
FITTING_ROOM_LAYERING_CONFIG_PATH=./config/layering_config.json
```

### How It Works

- **Custom config overrides defaults**: If you provide a config file, it merges with defaults
- **New garment types**: Add new types with appropriate layer numbers
- **Adjust existing**: Change layer numbers to reorder garments
- **Fallback**: If config file doesn't exist or fails to load, defaults are used

### Example: Custom Layering

```json
{
  "underwear": 0,
  "undershirt": 1,
  "vest": 2,        // Custom: vest goes under shirt
  "shirt": 3,
  "tie": 3.5,       // Custom: tie goes over shirt
  "jacket": 9,
  "pants": 6,
  "shoes": 12
}
```

## Configuration Priority

1. **Custom config file** (if `FITTING_ROOM_LAYERING_CONFIG_PATH` is set and file exists)
2. **Default layering order** (if no config file)

## Adding New Garment Types

When adding a new garment type:

1. **Add to config file** (if using custom config):
   ```json
   {
     "new_garment_type": 7.5
   }
   ```

2. **Or it will use default layer 100** (fitted last):
   - Unknown types are assigned layer 100
   - They'll be fitted after all known types

## Best Practices

1. **Use integers for main layers**: 0, 1, 2, 3...
2. **Use decimals for in-between**: 2.5, 3.5 (between 2 and 3, 3 and 4)
3. **Keep spacing**: Leave room between layers for future additions
4. **Document custom types**: Add comments in config or documentation

## Testing Layering

To verify layering order:

```python
from app.services.garment_fitting import GarmentFittingService

service = GarmentFittingService()
order = service.determine_layering_order(["shoes", "pants", "shirt", "underwear"])
print(order)  # Should show: [("underwear", 0), ("shirt", 3), ("pants", 6), ("shoes", 12)]
```
