<?php

declare(strict_types=1);

namespace guttedgarden\Tiktoken\Vocab;

use Countable;
use InvalidArgumentException;
use OutOfBoundsException;
use RuntimeException;
use guttedgarden\Tiktoken\Exception\ParseError;
use guttedgarden\Tiktoken\Util\EncodeUtil;

use function array_flip;
use function array_map;
use function assert;
use function base64_decode;
use function count;
use function explode;
use function fclose;
use function fgets;
use function file_exists;
use function fopen;
use function implode;
use function rewind;
use function sprintf;
use function stream_get_meta_data;
use function strval;

/** @psalm-import-type NonEmptyByteVector from EncodeUtil */
final class Vocab implements Countable
{
    /** @var array<non-empty-string, int> */
    private $tokenToRankMap;

    /** @var array<int, non-empty-string> */
    private $rankToTokenMap;

    /** @param array<non-empty-string, int> $tokenRankMap */
    private function __construct(array $tokenRankMap)
    {
        $this->tokenToRankMap = $tokenRankMap;
        /** @psalm-suppress PropertyTypeCoercion */
        $this->rankToTokenMap = array_map(static function ($value) {
            return strval($value);
        }, array_flip($tokenRankMap));

        if (count($this->tokenToRankMap) !== count($this->rankToTokenMap)) {
            throw new InvalidArgumentException('The map of tokens and ranks has duplicates of rank');
        }
    }

    /** @param non-empty-string $bpeFile */
    public static function fromFile(string $bpeFile): self
    {
        if (!file_exists($bpeFile)) {
            throw new RuntimeException(sprintf('File "%s" does not exist', $bpeFile));
        }

        $stream = fopen($bpeFile, 'rb');

        if ($stream === false) {
            throw new RuntimeException(sprintf('Could not open file: %s', $bpeFile));
        }

        try {
            return self::fromStream($stream);
        } finally {
            fclose($stream);
        }
    }

    /**
     * @param resource $stream
     *
     * @return self
     */
    public static function fromStream($stream): self
    {
        $meta = stream_get_meta_data($stream);

        if ($meta['seekable']) {
            rewind($stream);
        }

        $line = fgets($stream);
        $lineNo = 1;
        $map = [];

        while ($line !== false) {
            [$encodedToken, $rank] = explode(' ', $line);
            $token = base64_decode($encodedToken, true);

            if ($token === false) {
                throw new ParseError(sprintf('Could not decode token "%s" at line %d', $encodedToken, $lineNo));
            }

            assert($token !== '');

            $map[$token] = (int) $rank;

            $line = fgets($stream);
            $lineNo++;
        }

        return new self($map);
    }

    /**
     * @param string $binary
     * @return int|null
     */
    public function tryGetRank(string $binary): ?int
    {
        if ($binary === '') {
            throw new InvalidArgumentException('Argument $binary cannot be an empty string');
        }

        return $this->tokenToRankMap[$binary] ?? null;
    }

    /** @throws OutOfBoundsException */
    public function getRank(string $binary): int
    {
        if ($binary === '') {
            throw new InvalidArgumentException('Argument $binary cannot be an empty string');
        }

        if (!isset($this->tokenToRankMap[$binary])) {
            throw new OutOfBoundsException(sprintf(
                'No rank for bytes vector: [%s]',
                implode(', ', EncodeUtil::toBytes($binary))
            ));
        }

        return $this->tokenToRankMap[$binary];
    }

    /**
     * @param int $rank
     * @return string
     * @throws OutOfBoundsException
     */
    public function getToken(int $rank): string
    {
        if (!isset($this->rankToTokenMap[$rank])) {
            throw new OutOfBoundsException(sprintf('No token for rank: %d', $rank));
        }

        return $this->rankToTokenMap[$rank];
    }

    /** @psalm-api */
    public function count(): int
    {
        return count($this->tokenToRankMap);
    }
}
