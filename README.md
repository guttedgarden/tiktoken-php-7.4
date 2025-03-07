# tiktoken-php

[Yethee's tiktoken library](https://github.com/yethee/tiktoken-php) port for PHP 7.4 without the Symfony package

## Installation

```bash
$ composer require guttedgarden/tiktoken
```

## Usage

```php

use guttedgarden\Tiktoken\EncoderProvider;

$provider = new EncoderProvider();

$encoder = $provider->getForModel('gpt-3.5-turbo-0301');
$tokens = $encoder->encode('Hello world!');
print_r($tokens);
// OUT: [9906, 1917, 0]

$encoder = $provider->get('p50k_base');
$tokens = $encoder->encode('Hello world!');
print_r($tokens);
// OUT: [15496, 995, 0]
```

## Cache

The encoder uses an external vocabularies, so caching is used by default
to avoid performance issues.

By default, the [directory for temporary files](https://www.php.net/manual/en/function.sys-get-temp-dir.php) is used.
You can override the directory for cache via environment variable `TIKTOKEN_CACHE_DIR`
or use `EncoderProvider::setVocabCache()`:

```php
use guttedgarden\Tiktoken\EncoderProvider;

$encProvider = new EncoderProvider();
$encProvider->setVocabCache('/path/to/cache');

// Using the provider
```

### Disable cache

You can disable the cache, if there are reasons for this,
in one of the following ways:

* Set an empty string for the environment variable `TIKTOKEN_CACHE_DIR`.
* Programmatically:

```php
use guttedgarden\Tiktoken\EncoderProvider;

$encProvider = new EncoderProvider();
$encProvider->setVocabCache(null); // disable the cache
```

## Limitations

* Encoding for GPT-2 is not supported.
* Special tokens (like `<|endofprompt|>`) are not supported.

## License

[MIT](./LICENSE)
