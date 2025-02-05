<?php

declare(strict_types=1);

namespace guttedgarden\Tiktoken\Vocab;

interface VocabLoader
{
    /**
     * @param string $uri
     * @param string|null $checksum
     * @return Vocab
     */
    public function load(string $uri, ?string $checksum = null): Vocab;
}
