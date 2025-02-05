<?php

declare(strict_types=1);

namespace guttedgarden\Tiktoken;

use guttedgarden\Tiktoken\Exception\RegexError;
use guttedgarden\Tiktoken\Util\EncodeUtil;
use guttedgarden\Tiktoken\Vocab\Vocab;

use function array_map;
use function array_merge;
use function array_values;
use function count;
use function implode;
use function preg_last_error;
use function preg_match_all;
use function sprintf;
use function strlen;
use function substr;

use const PHP_INT_MAX;

/** @psalm-import-type NonEmptyByteVector from EncodeUtil */
final class Encoder
{
    /**
     * @var string
     */
    private $name;

    /**
     * @var Vocab
     */
    private $vocab;

    /**
     * @var string
     */
    private $pattern;

    /**
     * @param string $name
     * @param string $pattern
     */
    public function __construct(string $name, Vocab $vocab, string $pattern)
    {
        $this->name = $name;
        $this->vocab = $vocab;
        $this->pattern = $pattern;
    }

    public function __toString(): string
    {
        return sprintf('Encoder(name="%s", vocab=%d)', $this->name, count($this->vocab));
    }

    /** @return int[] */
    public function encode(string $text): array
    {
        if ($text === '') {
            return [];
        }

        if (preg_match_all($this->pattern, $text, $matches) === false) {
            throw new RegexError(sprintf('Matching failed with error code: %d', preg_last_error()));
        }

        $tokens = [];

        foreach ($matches[0] as $match) {
            if ($match === '') {
                continue;
            }

            $rank = $this->vocab->tryGetRank($match);

            if ($rank !== null) {
                $tokens[] = $rank;
                continue;
            }

            foreach ($this->mergeBytePairs($match) as $rank) {
                $tokens[] = $rank;
            }
        }

        return $tokens;
    }

    /** @return int[][] */
    public function encodeInChunks(string $text, int $maxTokensPerChunk): array
    {
        if ($text === '') {
            return [];
        }

        if (preg_match_all($this->pattern, $text, $matches) === false) {
            throw new RegexError(sprintf('Matching failed with error code: %d', preg_last_error()));
        }

        $chunks = [];
        $tokensInCurrentChunk = [];

        foreach ($matches[0] as $match) {
            if ($match === '') {
                continue;
            }

            $rank = $this->vocab->tryGetRank($match);
            $tokens = $rank !== null ? [$rank] : $this->mergeBytePairs($match);

            if (count($tokensInCurrentChunk) + count($tokens) > $maxTokensPerChunk) {
                $chunks[] = $tokensInCurrentChunk;
                $tokensInCurrentChunk = [];
            }

            $tokensInCurrentChunk = array_merge($tokensInCurrentChunk, $tokens);
        }

        if (count($tokensInCurrentChunk) > 0) {
            $chunks[] = $tokensInCurrentChunk;
        }

        return $chunks;
    }

    /** @param int[] $tokens */
    public function decode(array $tokens): string
    {
        if ($tokens === []) {
            return '';
        }

        return implode(array_map([$this->vocab, 'getToken'], $tokens));
    }

    /**
     * @param string $piece
     *
     * @return int[]
     */
    private function mergeBytePairs(string $piece): array
    {
        $parts = [];

        for ($i = 0; $i <= strlen($piece); $i++) {
            $parts[] = [$i, PHP_INT_MAX];
        }

        $getRank = function (array $parts, int $startIndex, int $skip = 0) use ($piece): int {
            if (($startIndex + $skip + 2) >= count($parts)) {
                return PHP_INT_MAX;
            }

            $offset = $parts[$startIndex][0];
            $length = $parts[$startIndex + $skip + 2][0] - $offset;

            return $this->vocab->tryGetRank(substr($piece, $offset, $length)) ?? PHP_INT_MAX;
        };

        for ($i = 0; $i < count($parts) - 2; $i++) {
            $parts[$i][1] = $getRank($parts, $i);
        }

        while (count($parts) > 1) {
            $minRank = PHP_INT_MAX;
            $partIndex = 0;
            $stop = count($parts) - 1;

            for ($i = 0; $i < $stop; $i++) {
                if ($minRank <= $parts[$i][1]) {
                    continue;
                }

                $minRank = $parts[$i][1];
                $partIndex = $i;
            }

            if ($minRank === PHP_INT_MAX) {
                break;
            }

            unset($parts[$partIndex + 1]);
            $parts = array_values($parts);

            $parts[$partIndex][1] = $getRank($parts, $partIndex);

            if ($partIndex <= 0) {
                continue;
            }

            $parts[$partIndex - 1][1] = $getRank($parts, $partIndex - 1);
        }

        $stop = count($parts) - 1;
        $res = [];

        for ($i = 0; $i < $stop; $i++) {
            $offset = $parts[$i][0];
            $length = $parts[$i + 1][0] - $offset;

            $res[] = $this->vocab->getRank(substr($piece, $offset, $length));
        }

        return $res;
    }
}
