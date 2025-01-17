<?php

declare(strict_types=1);

namespace guttedgarden\Tiktoken;

use Closure;
use guttedgarden\Tiktoken\Exception\RegexError;
use guttedgarden\Tiktoken\Util\EncodeUtil;
use guttedgarden\Tiktoken\Vocab\Vocab;

use function array_map;
use function array_slice;
use function array_values;
use function assert;
use function count;
use function implode;
use function preg_match_all;
use function range;
use function sprintf;

use const PHP_INT_MAX;

/** @psalm-import-type NonEmptyByteVector from EncodeUtil */
final class Encoder
{
    private string $name;
    private Vocab $vocab;
    private string $pattern;
    /**
     * @param non-empty-string $name
     * @param non-empty-string $pattern
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

    /** @return list<int> */
    public function encode(string $text): array
    {
        if ($text === '') {
            return [];
        }

        if (preg_match_all($this->pattern, $text, $matches) === false) {
            throw new RegexError(sprintf('Matching failed with error: %s', $this->pregErrorString(preg_last_error())));
        }

        $tokens = [];

        foreach ($matches[0] as $match) {
            if ($match === '') {
                continue;
            }

            $piece = EncodeUtil::toBytes($match);
            $rank = $this->vocab->tryGetRank($piece);

            if ($rank !== null) {
                $tokens[] = $rank;

                continue;
            }

            foreach ($this->mergeBytePairs($piece) as $rank) {
                $tokens[] = $rank;
            }
        }

        return $tokens;
    }

    /** @param array<int> $tokens */
    public function decode(array $tokens): string
    {
        if ($tokens === []) {
            return '';
        }

        return implode(array_map(Closure::fromCallable([$this->vocab, 'getToken']), $tokens));
    }

    /**
     * @psalm-param NonEmptyByteVector $bytes
     *
     * @return list<int>
     */
    private function mergeBytePairs(array $bytes): array
    {
        /** @var list<array{int, int}> $parts */
        $parts = array_map(
            function (int $i) use ($bytes): array {
                if ($i + 1 < count($bytes)) {
                    $piece = array_slice($bytes, $i, 2);
                    assert(count($piece) === 2);

                    return [$i, $this->vocab->tryGetRank($piece) ?? PHP_INT_MAX];
                }

                return [$i, PHP_INT_MAX];
            },
            range(0, count($bytes)),
        );
        $getRank = function (array $parts, int $startIndex) use ($bytes): int {
            if ($startIndex + 2 >= count($parts)) {
                return PHP_INT_MAX;
            }

            $offset = $parts[$startIndex][0];
            $piece  = array_slice($bytes, $offset, $parts[$startIndex + 2][0] - $offset);
            assert(count($piece) > 0);

            return $this->vocab->tryGetRank($piece) ?? PHP_INT_MAX;
        };

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
            $piece = array_slice($bytes, $parts[$i][0], $parts[$i + 1][0] - $parts[$i][0]);
            assert(count($piece) > 0);

            $res[] = $this->vocab->getRank($piece);
        }

        return $res;
    }

    private function pregErrorString($errorConstant)
    {
        static $errorMessages = [
            PREG_NO_ERROR            => 'No error',
            PREG_INTERNAL_ERROR      => 'Internal error',
            PREG_BACKTRACK_LIMIT_ERROR=> 'Backtrack limit error',
            PREG_RECURSION_LIMIT_ERROR=> 'Recursion limit error',
            PREG_BAD_UTF8_ERROR       => 'Bad UTF8 error',
            PREG_BAD_UTF8_OFFSET_ERROR=> 'Bad UTF8 offset error',
        ];

        return $errorMessages[$errorConstant] ?? 'Unknown error';
    }
}
