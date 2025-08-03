# import asyncio
# from pymodbus.server import StartAsyncTcpServer
# from pymodbus.datastore import ModbusServerContext, ModbusSlaveContext, ModbusSequentialDataBlock
# from pymodbus.device import ModbusDeviceIdentification


# def on_write_coil(address, value):
#     """
#     This function will be called whenever a client writes to a coil.
#     It prints the address of the coil and the new value.
#     """
#     print(f"✅ Client successfully wrote to coil at address {address}. New value is: {value}")
# # ----------------------------
# # Create datastore
# # ----------------------------
# store = ModbusSlaveContext(
#     di=ModbusSequentialDataBlock(0, [17]*100),
#     co=ModbusSequentialDataBlock(0, [True]*100, on_set=on_write_coil),
#     hr=ModbusSequentialDataBlock(0, [123]*100),
#     ir=ModbusSequentialDataBlock(0, [456]*100)
# )

# context = ModbusServerContext(slaves=store, single=True)

# # ----------------------------
# # Device identity
# # ----------------------------
# identity = ModbusDeviceIdentification()
# identity.VendorName = "ChatGPT Async Modbus"
# identity.ProductCode = "CGPM"
# identity.VendorUrl = "https://openai.com"
# identity.ProductName = "Modbus TCP Async Server"
# identity.ModelName = "ModbusServerModel"
# identity.MajorMinorRevision = "3.8.6"

# # ----------------------------
# # Start Async Server
# # ----------------------------
# async def run_server():
#     print("🟢 Starting Modbus TCP Async Server on localhost:5020")
#     await StartAsyncTcpServer(
#         context,
#         identity=identity,
#         address=("0.0.0.0", 5020)
#     )

# if __name__ == "__main__":
#     asyncio.run(run_server())


import asyncio
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusServerContext, ModbusSlaveContext, ModbusSequentialDataBlock
from pymodbus.device import ModbusDeviceIdentification

# ----------------------------
# Custom Data Block with Callback
# ----------------------------
class MyCoilDataBlock(ModbusSequentialDataBlock):
    """A custom data block that prints a message on write."""
    def setValues(self, address, values):
        """
        This method is automatically called when a client writes to the coils.
        """
        for i, value in enumerate(values):
            print(f"✅ Client successfully wrote to coil at address {address + i}. New value is: {value}")
        # Call the original setValues method to actually store the new values
        super().setValues(address, values)

# ----------------------------
# Create datastore with the custom data block
# ----------------------------
store = ModbusSlaveContext(
    di=ModbusSequentialDataBlock(0, [17]*100),
    co=MyCoilDataBlock(0, [True]*100),  # <-- Use your custom data block here
    hr=ModbusSequentialDataBlock(0, [123]*100),
    ir=ModbusSequentialDataBlock(0, [456]*100)
)

context = ModbusServerContext(slaves=store, single=True)

# ----------------------------
# Device identity
# ----------------------------
identity = ModbusDeviceIdentification()
identity.VendorName = "ChatGPT Async Modbus"
identity.ProductCode = "CGPM"
identity.VendorUrl = "https://openai.com"
identity.ProductName = "Modbus TCP Async Server"
identity.ModelName = "ModbusServerModel"
identity.MajorMinorRevision = "3.8.6"

# ----------------------------
# Start Async Server
# ----------------------------
async def run_server():
    print("🟢 Starting Modbus TCP Async Server on 0.0.0.0:5020")
    await StartAsyncTcpServer(
        context,
        identity=identity,
        address=("0.0.0.0", 5020)
    )

if __name__ == "__main__":
    asyncio.run(run_server())