<?php

declare(strict_types=1);

namespace guttedgarden\Tiktoken\Util;

use Closure;
use function array_map;
use function bin2hex;
use function pack;
use function str_split;

/** @psalm-type NonEmptyByteVector = non-empty-list<int<0, 255>> */
final class EncodeUtil
{
    /**
     * @param non-empty-string $text Text must be valid UTF-8 string.
     *
     * @psalm-return NonEmptyByteVector
     */
    public static function toBytes(string $text): array
    {
        return array_map(Closure::fromCallable('hexdec'), str_split(bin2hex($text), 2));
    }

    /**
     * @psalm-param NonEmptyByteVector $bytes
     *
     * @return non-empty-string
     */
    public static function fromBytes(array $bytes): string
    {
        return pack('C*', ...$bytes);
    }
}
