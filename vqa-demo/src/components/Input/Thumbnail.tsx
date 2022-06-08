import { Box, Image, Spinner } from '@chakra-ui/react'

type ImageThumbProps = {
  imageSrc: string
  loading: boolean
}

function Thumbnail({
  imageSrc,
  loading,
}: ImageThumbProps) {
  if (loading) {
    return (
      <Spinner
        thickness='4px'
        speed='0.65s'
        emptyColor='gray.200'
        color='red.500'
        size='xl'
      />
    )
  }

  return (
    <Box>
      { imageSrc && (
        <Image
          src={imageSrc}
          boxSize='lg'
          alt="preview-img"
        />
      )}
    </Box>
  )
}

export default Thumbnail
