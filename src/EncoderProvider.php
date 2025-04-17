<?php

declare(strict_types=1);

namespace guttedgarden\Tiktoken;

use InvalidArgumentException;
use guttedgarden\Tiktoken\Vocab\Loader\DefaultVocabLoader;
use guttedgarden\Tiktoken\Vocab\Vocab;
use guttedgarden\Tiktoken\Vocab\VocabLoader;

use function getenv;
use function sprintf;
use function sys_get_temp_dir;

use const DIRECTORY_SEPARATOR;

final class EncoderProvider
{
    /** @var array<string, array<string, string>> */
    private const ENCODINGS = [
        'r50k_base' => [
            'vocab' => 'https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken',
            'hash' => '306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930',
            'pat' => '/\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/u',
        ],
        'p50k_base' => [
            'vocab' => 'https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken',
            'hash' => '94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069',
            'pat' => '/\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/u',
        ],
        'p50k_edit' => [
            'vocab' => 'https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken',
            'hash' => '94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069',
            'pat' => '/\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/u',
        ],
        'cl100k_base' => [
            'vocab' => 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken',
            'hash' => '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7',
            'pat' => '/(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/u',
        ],
        'o200k_base' => [
            'vocab' => 'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken',
            'hash' => '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d',
            'pat' => '/[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n\/]*|\s*[\r\n]+|\s+(?!\S)|\s+/u',
        ],
    ];

    /** @var array<string, string> */
    private const MODEL_PREFIX_TO_ENCODING = [
        'gpt-4.5-' => 'o200k_base',
        'o3-' => 'o200k_base',
        'o1-' => 'o200k_base',
        'chatgpt-4o-' => 'o200k_base',
        'gpt-4o-' => 'o200k_base',
        'gpt-4-' => 'cl100k_base',
        'gpt-3.5-turbo-' => 'cl100k_base',
        'gpt-4.1-' => 'o200k_base',
        'o4-mini-' => 'o200k_base',
    ];

    /** @var array<string, string> */
    private const MODEL_TO_ENCODING = [
        'gpt-4.5' => 'o200k_base',
        'o3' => 'o200k_base',
        'o1' => 'o200k_base',
        'gpt-4o' => 'o200k_base',
        'gpt-4' => 'cl100k_base',
        'gpt-3.5-turbo' => 'cl100k_base',
        'gpt-3.5' => 'cl100k_base',
        'gpt-4.1' => 'o200k_base',
        'gpt-4.1-mini' => 'o200k_base',
        'gpt-4.1-nano' => 'o200k_base',
        'o1-pro' => 'o200k_base',
        'o4-mini' => 'o200k_base',
        'davinci-002' => 'cl100k_base',
        'babbage-002' => 'cl100k_base',
        'text-embedding-ada-002' => 'cl100k_base',
        'text-embedding-3-small' => 'cl100k_base',
        'text-embedding-3-large' => 'cl100k_base',
        'text-davinci-003' => 'p50k_base',
        'text-davinci-002' => 'p50k_base',
        'text-davinci-001' => 'r50k_base',
        'text-curie-001' => 'r50k_base',
        'text-babbage-001' => 'r50k_base',
        'text-ada-001' => 'r50k_base',
        'davinci' => 'r50k_base',
        'curie' => 'r50k_base',
        'babbage' => 'r50k_base',
        'ada' => 'r50k_base',
        'code-davinci-002' => 'p50k_base',
        'code-davinci-001' => 'p50k_base',
        'code-cushman-002' => 'p50k_base',
        'code-cushman-001' => 'p50k_base',
        'davinci-codex' => 'p50k_base',
        'cushman-codex' => 'p50k_base',
        'text-davinci-edit-001' => 'p50k_edit',
        'code-davinci-edit-001' => 'p50k_edit',
        'text-similarity-davinci-001' => 'r50k_base',
        'text-similarity-curie-001' => 'r50k_base',
        'text-similarity-babbage-001' => 'r50k_base',
        'text-similarity-ada-001' => 'r50k_base',
        'text-search-davinci-doc-001' => 'r50k_base',
        'text-search-curie-doc-001' => 'r50k_base',
        'text-search-babbage-doc-001' => 'r50k_base',
        'text-search-ada-doc-001' => 'r50k_base',
        'code-search-babbage-code-001' => 'r50k_base',
        'code-search-ada-code-001' => 'r50k_base',
    ];

    /** @var VocabLoader|null */
    private $vocabLoader = null;

    /** @var string|null */
    private $vocabCacheDir;

    /** @var array<string, Encoder> */
    private $encoders = [];

    /** @var array<string, Vocab> */
    private $vocabs = [];

    public function __construct()
    {
        $cacheDir = getenv('TIKTOKEN_CACHE_DIR');

        if ($cacheDir === false) {
            $cacheDir = sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'tiktoken';
        }

        $this->vocabCacheDir = $cacheDir !== '' ? $cacheDir : null;
    }

    /**
     * @param string $model
     * @return Encoder
     * @throws InvalidArgumentException
     */
    public function getForModel(string $model): Encoder
    {
        if (isset(self::MODEL_TO_ENCODING[$model])) {
            return $this->get(self::MODEL_TO_ENCODING[$model]);
        }

        foreach (self::MODEL_PREFIX_TO_ENCODING as $prefix => $modelEncoding) {
            if (strpos($model, $prefix) === 0) {
                return $this->get($modelEncoding);
            }
        }

        throw new InvalidArgumentException(sprintf('Unknown model name: %s', $model));
    }

    /**
     * @param string $encodingName
     * @return Encoder
     * @throws InvalidArgumentException
     */
    public function get(string $encodingName): Encoder
    {
        if (!isset(self::ENCODINGS[$encodingName])) {
            throw new InvalidArgumentException(sprintf('Unknown encoding: %s', $encodingName));
        }

        if (!isset($this->encoders[$encodingName])) {
            $options = self::ENCODINGS[$encodingName];

            return $this->encoders[$encodingName] = new Encoder(
                $encodingName,
                $this->getVocab($encodingName),
                $options['pat']
            );
        }

        return $this->encoders[$encodingName];
    }

    /**
     * @param string|null $cacheDir
     * @return void
     */
    public function setVocabCache(?string $cacheDir): void
    {
        $this->vocabCacheDir = $cacheDir;
        $this->vocabLoader = null;
    }

    /**
     * @param VocabLoader $loader
     * @return void
     */
    public function setVocabLoader(VocabLoader $loader): void
    {
        $this->vocabLoader = $loader;
    }

    /**
     * @return void
     */
    public function reset(): void
    {
        $this->encoders = [];
        $this->vocabs = [];
    }

    /**
     * @param string $encodingName
     * @return Vocab
     */
    private function getVocab(string $encodingName): Vocab
    {
        if (isset($this->vocabs[$encodingName])) {
            return $this->vocabs[$encodingName];
        }

        $loader = $this->vocabLoader;

        if ($loader === null) {
            $loader = $this->vocabLoader = new DefaultVocabLoader($this->vocabCacheDir);
        }

        return $this->vocabs[$encodingName] = $loader->load(
            self::ENCODINGS[$encodingName]['vocab'],
            isset(self::ENCODINGS[$encodingName]['hash']) ? self::ENCODINGS[$encodingName]['hash'] : null
        );
    }
}
