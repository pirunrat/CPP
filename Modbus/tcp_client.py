# TCP/tcp_client.py
from pymodbus.client import AsyncModbusTcpClient
from Utils.Interface import ModbusClientInterface
import asyncio


class ModbusTcpClient(ModbusClientInterface):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._client = None



    async def connect(self):
        self._client = AsyncModbusTcpClient(self._host, self._port)
        is_connected = await self._client.connect()
        if not is_connected:
            self._client = None
        return is_connected



    async def read_coils(self, address: int, count: int, unit: int = 1):
        if self._client is None:
            print("[ERROR] Client not connected.")
            return None
        result = await self._client.read_coils(address, count, unit=unit)
        return result.bits if result and not result.isError() else None



    async def write_coil(self, address: int, value: bool, unit: int = 1):
        if self._client is None:
            print("[ERROR] Client not connected.")
            return False
        result = await self._client.write_coil(address, value, unit=unit)
        return result and not result.isError()
    


    # async def write_register(self, address: int, value: int, unit: int = 1):
    #     if self._client is None:
    #         print("[ERROR] Client not connectd.")
    #         return False
    #     result = await self._client.write_register(address, value, unit=unit)
    #     return result and not result.isError()



    async def write_register(self, address: int, value, unit: int = 1):
        if self._client is None:
            print("[ERROR] Client not connected.")
            return False

        try:
            if isinstance(value, list):
                result = await self._client.write_registers(address, value, unit=unit)
            else:
                result = await self._client.write_register(address, value, unit=unit)

            return result and not result.isError()

        except Exception as e:
            print(f"[EXCEPTION] write_register error: {e}")
            return False

    


    async def disconnect(self):
        if self._client:
            close_method = getattr(self._client, "close", None)
            if callable(close_method):
                maybe_awaitable = close_method()
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            self._client = None