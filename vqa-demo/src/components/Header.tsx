/* External dependencies */
import * as React from 'react'
import {
  Box,
  Flex,
  Heading,
  useColorModeValue,
  useBreakpointValue,
} from '@chakra-ui/react'

/* Internal dependencies */
import { ColorModeSwitcher } from '../ColorModeSwitcher'

function Header() {
  return (
    <Box
      borderBottom={1}
      borderStyle='solid'
      borderColor={useColorModeValue('gray.200', 'gray.900')}
      px={{ base: 4 }}
    >
      <Flex
        bg={useColorModeValue('white', 'gray.800')}
        color={useColorModeValue('gray.600', 'white')}
        minH='60px'
        mx={{ base: 118 }}
        align='center'
      >
        <Flex
          flex={{ base: 1 }}
          justify={{ base: 'center', md: 'start' }}
        >
          <Heading
            as='h4'
            size='md'
            textAlign={useBreakpointValue({ base: 'center', md: 'left' })}
            color={useColorModeValue('gray.700', 'white')}>
            Korea University COSE461 Final Project - Team 12
          </Heading>
        </Flex>
        <ColorModeSwitcher justifySelf="flex-end" />
      </Flex>
    </Box>
  )
}

export default Header
